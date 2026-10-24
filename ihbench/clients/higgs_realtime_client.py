"""Boson Higgs Realtime API client (text-only output) over WebSocket.

The server speaks an OpenAI-Realtime-like protocol with a few differences that
matter for replaying a conversation history:

* nothing is sent until the first ``session.update``; ``session.created`` acks it
* ``conversation.item.create`` accepts text only, so user audio goes through
  ``input_audio_buffer.append`` + ``commit`` (manual turns, ``turn_detection: null``)
* ``conversation.item.done`` is never emitted, so items are not awaited
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

import websockets
import websockets.exceptions

from ihbench.clients.base import (
    APIClient,
    GenerationConfig,
    GenericMessage,
    is_audio_content,
)
from ihbench.clients.openai_realtime_client import _wav_to_pcm24k_b64

URL = "wss://api.boson.ai/v1/realtime"
ATTEMPT_TIMEOUT_S = 60.0
CLOSE_TIMEOUT_S = 1.0  # bounds the close handshake so a timed-out attempt ends promptly (AC-21)
LOG_PREFIX = "higgs-realtime attempt failed:"
MAX_APPEND_B64_CHARS = 1_048_576  # 1 MiB of base64 per input_audio_buffer.append
_KEY_ENV = "BOSON_API_KEY"
_KEY_PLACEHOLDER = "${BOSON_API_KEY}"

_log = logging.getLogger(__name__)


class HiggsRealtimeError(RuntimeError):
    """A failed attempt. Messages carry event types, codes and statuses, never payloads or keys."""


def _headers_kwarg(ws_version: str) -> str:
    """Name of the ``websockets.connect`` keyword for request headers.

    The legacy client (12.x-13.x) takes ``extra_headers``; the new client
    (14.0+) takes ``additional_headers``.
    """
    major = int(ws_version.split(".")[0])
    return "additional_headers" if major >= 14 else "extra_headers"


def _chunk_b64(b64: str, limit: int = MAX_APPEND_B64_CHARS) -> list[str]:
    """Split base64 text into pieces of at most ``limit`` chars, each a multiple of 4."""
    size = limit - (limit % 4)
    return [b64[i:i + size] for i in range(0, len(b64), size)]


class HiggsRealtimeClient(APIClient):
    """Client for Higgs Realtime. Audio is sent as 24 kHz mono PCM16, output is text."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        attempt_timeout_s: float = ATTEMPT_TIMEOUT_S,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        key = api_key if api_key is not None else os.environ.get(_KEY_ENV)
        if key is None or not key.strip() or key.strip() == _KEY_PLACEHOLDER:
            raise ValueError(f"{_KEY_ENV} is missing, empty or an unexpanded placeholder")
        self._api_key = key
        self.attempt_timeout_s = attempt_timeout_s

    def __repr__(self) -> str:
        return f"HiggsRealtimeClient(model_name={self.model_name!r})"

    # ------------------------------------------------------------------ building

    def _session_update(self, system_message: str, config: GenerationConfig) -> dict:
        return {
            "type": "session.update",
            "session": {
                "model": self.model_name,
                "output_modalities": ["text"],
                "instructions": system_message,
                "max_output_tokens": config.max_new_tokens,
                "audio": {"input": {"turn_detection": None}},
            },
        }

    def _build_replay(self, messages: list[GenericMessage]) -> list[tuple[str, Any]]:
        """Turn the history into ("audio", [b64 chunks]) and ("item", dict) steps."""
        steps: list[tuple[str, Any]] = []
        for i, msg in enumerate(messages):
            if msg.type == "audio" and is_audio_content(msg.content):
                b64 = _wav_to_pcm24k_b64(msg.content)
                if not b64:
                    raise HiggsRealtimeError(f"{LOG_PREFIX} user turn {i} audio has zero samples")
                steps.append(("audio", _chunk_b64(b64)))
            elif msg.role == "user":
                steps.append(("item", {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content}],
                }))
            elif msg.role == "assistant" and msg.content:
                steps.append(("item", {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg.content}],
                }))
        return steps

    # ------------------------------------------------------------------ wire

    @staticmethod
    def _closed_error(
        e: websockets.exceptions.ConnectionClosed, before: str, state: dict
    ) -> HiggsRealtimeError:
        """Failure for a closed socket. Names the close code only, never the reason text."""
        terminal = state.get("terminal")
        if terminal:
            cause = f"after {terminal}"
        elif e.rcvd is not None:
            cause = f"close code {e.rcvd.code}"
        else:
            cause = "no close frame"
        return HiggsRealtimeError(f"{LOG_PREFIX} connection closed before {before} ({cause})")

    async def _send(self, ws: Any, event: dict, before: str, state: dict) -> None:
        try:
            await ws.send(json.dumps(event))
        except websockets.exceptions.ConnectionClosed as e:
            raise self._closed_error(e, before, state) from None

    async def _wait_for(self, ws: Any, event_type: str, state: dict) -> dict:
        """Read events until one of ``event_type`` arrives; ignore unknown ones."""
        while True:
            try:
                raw = await ws.recv()
            except websockets.exceptions.ConnectionClosed as e:
                raise self._closed_error(e, event_type, state) from None
            try:
                event = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "error":
                err = event.get("error")
                err = err if isinstance(err, dict) else {}
                detail = str(err.get("type"))
                if err.get("code") is not None:
                    detail += f" ({err['code']})"
                raise HiggsRealtimeError(f"{LOG_PREFIX} error event {detail}")
            if etype in ("session.idle_timeout", "session.max_duration_reached"):
                state["terminal"] = etype
            elif etype == "response.output_text.delta" and state.get("collect"):
                state["text"].append(str(event.get("delta", "")))
            if etype == event_type:
                return event

    async def _run(
        self,
        system_message: str,
        steps: list[tuple[str, Any]],
        config: GenerationConfig,
    ) -> str:
        kwarg = _headers_kwarg(importlib.metadata.version("websockets"))
        headers = {"Authorization": f"Bearer {self._api_key}"}
        state: dict[str, Any] = {"text": [], "collect": False}
        async with AsyncExitStack() as stack:
            try:
                ws = await stack.enter_async_context(websockets.connect(URL, close_timeout=CLOSE_TIMEOUT_S, **{kwarg: headers}))
            except Exception as e:
                raise HiggsRealtimeError(f"{LOG_PREFIX} connection failed: {type(e).__name__}") from None

            await self._send(ws, self._session_update(system_message, config), "session.created", state)
            await self._wait_for(ws, "session.created", state)  # never session.updated (F14)

            for kind, payload in steps:
                if kind == "audio":
                    for chunk in payload:
                        await self._send(
                            ws, {"type": "input_audio_buffer.append", "audio": chunk},
                            "input_audio_buffer.committed", state,
                        )
                    await self._send(ws, {"type": "input_audio_buffer.commit"}, "input_audio_buffer.committed", state)
                    await self._wait_for(ws, "input_audio_buffer.committed", state)
                else:
                    await self._send(
                        ws, {"type": "conversation.item.create", "item": payload}, "response.done", state
                    )

            state["collect"] = True
            await self._send(ws, {"type": "response.create"}, "response.done", state)
            done = await self._wait_for(ws, "response.done", state)

        response = done.get("response")
        status = response.get("status") if isinstance(response, dict) else None
        if status != "completed":
            raise HiggsRealtimeError(f"{LOG_PREFIX} response {status if isinstance(status, str) else 'status missing'}")
        return "".join(state["text"])

    async def _generate_async(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig,
    ) -> str:
        try:
            steps = self._build_replay(messages)  # before connecting: a bad turn costs nothing
            try:
                return await asyncio.wait_for(
                    self._run(system_message, steps, config), self.attempt_timeout_s
                )
            except asyncio.TimeoutError:
                raise HiggsRealtimeError(
                    f"{LOG_PREFIX} no response.done within {self.attempt_timeout_s:g}s (timeout)"
                ) from None
        except HiggsRealtimeError as err:
            _log.warning("%s", err)
            raise

    def generate(
        self,
        system_message: str,
        messages: list[GenericMessage],
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()
        return asyncio.run(self._generate_async(system_message, messages, config))
