#!/usr/bin/env python3
"""Run inference on IHBench samples using an API-backed model.

Generates model responses for all evaluation samples and saves them to a JSONL
file. These responses can then be evaluated using run_evaluate.py.

Usage:
    python scripts/run_inference_api.py \
        --dataset /path/to/verified_conversations.jsonl \
        --backend openai \
        --model gpt-4o-audio-preview \
        --output responses/gpt-4o-audio-preview.jsonl \
        --max-workers 8

Backends: openai, openai-realtime, gemini, vllm
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from eval/ directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ihbench.clients.base import GenerationConfig, GenericMessage
from ihbench.data import EvalSample, load_eval_samples, load_eval_samples_from_hf


def make_client(args):
    if args.backend == "openai":
        from ihbench.clients.openai_client import OpenAIChatClient

        return OpenAIChatClient(model_name=args.model, api_key=args.api_key, base_url=args.base_url, reasoning_effort=getattr(args, "reasoning_effort", None))
    elif args.backend == "openai-realtime":
        from ihbench.clients.openai_realtime_client import OpenAIRealtimeClient

        return OpenAIRealtimeClient(model_name=args.model, api_key=args.api_key, reasoning_effort=getattr(args, "reasoning_effort", None))
    elif args.backend == "higgs-realtime":
        from ihbench.clients.higgs_realtime_client import HiggsRealtimeClient

        return HiggsRealtimeClient(model_name=args.model, api_key=args.api_key)
    elif args.backend == "gemini":
        from ihbench.clients.gemini_client import GeminiChatClient

        return GeminiChatClient(
            model_name=args.model,
            api_key=args.api_key,
            thinking_budget=args.thinking_budget,
        )
    elif args.backend == "gemini-live":
        from ihbench.clients.gemini_live_client import GeminiLiveClient

        return GeminiLiveClient(model_name=args.model, api_key=args.api_key, include_thoughts=getattr(args, "include_thoughts", None))
    elif args.backend == "vllm":
        from ihbench.clients.vllm_client import VLLMClient

        return VLLMClient(
            model_name=args.model,
            base_url=args.base_url or "http://localhost:8000/v1",
            served_model_name=args.served_model_name,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
def _generate_with_retry(client, system_message, messages, config):
    return client.generate(system_message, messages, config)


def run_sample(client, sample: EvalSample, config: GenerationConfig, audio_dir: str | None = None) -> dict:
    """Generate a response for a single sample."""
    messages = []
    for m in sample.format_history_for_model(audio_dir=audio_dir):
        if "audio_path" in m:
            messages.append(GenericMessage(role=m["role"], type="audio", content=m["audio_path"]))
        else:
            messages.append(GenericMessage(role=m["role"], type="text", content=m["content"]))
    response = _generate_with_retry(client, sample.system_message, messages, config)
    result = {
        "sample_id": sample.id,
        "conversation_id": sample.conversation_id,
        "interruption_type": sample.interruption_type,
        "turn_depth": sample.turn_depth,
        "response": response,
    }
    if hasattr(sample, "epoch"):
        result["epoch"] = sample.epoch
    return result


def main():
    parser = argparse.ArgumentParser(description="Run IHBench inference with an API model")
    parser.add_argument("--dataset", default=None,
                        help="Path to a local conversations JSONL. If omitted, the "
                             "public HuggingFace dataset (--hf-repo) is used.")
    parser.add_argument("--hf-repo", default="bosonai/ihbench",
                        help="HuggingFace dataset repo (used when --dataset is omitted)")
    parser.add_argument("--audio-cache", default=".ihbench_audio_cache",
                        help="Local dir where HF-embedded audio is materialized")
    parser.add_argument("--backend", default=None, choices=["openai", "openai-realtime", "higgs-realtime", "gemini", "gemini-live", "vllm"])
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--output", required=True, help="Output JSONL path for responses")
    parser.add_argument("--api-key", default=None, help="API key (defaults to env var)")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI/vLLM")
    parser.add_argument("--served-model-name", default=None, help="Served model name for vLLM")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Thinking budget for Gemini")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for API calls")
    parser.add_argument("--epochs", type=int, default=1, help="Number of inference epochs (for CI)")
    parser.add_argument("--config", default=None, help="YAML model config file")
    parser.add_argument("--audio-dir", default=None, help="Directory with extracted WAV files (data/audio/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    # YAML config overrides
    if args.config:
        import os
        import re
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Expand ${VAR} patterns in string values
        for k, v in cfg.items():
            if isinstance(v, str):
                cfg[k] = re.sub(
                    r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)",
                    lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
                    v,
                )
        for key in ("backend", "model", "api_key", "base_url", "served_model_name",
                     "thinking_budget", "include_thoughts", "reasoning_effort", "temperature", "max_new_tokens", "max_workers"):
            if getattr(args, key, None) is None and key in cfg:
                setattr(args, key, cfg[key])

    if not args.backend or not args.model:
        parser.error("--backend and --model are required (via CLI or YAML config)")

    if args.dataset:
        print(f"Loading dataset from {args.dataset} (epochs={args.epochs})...", file=sys.stderr)
        samples = load_eval_samples(args.dataset, epochs=args.epochs)
        audio_dir = args.audio_dir
    else:
        print(f"Loading HuggingFace dataset {args.hf_repo} (epochs={args.epochs})...", file=sys.stderr)
        samples = load_eval_samples_from_hf(args.hf_repo, args.audio_cache, epochs=args.epochs)
        audio_dir = args.audio_cache  # HF loader materialized audio here
    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} evaluation samples", file=sys.stderr)

    client = make_client(args)
    config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_new_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0

    with open(output_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(run_sample, client, s, config, audio_dir): s for s in samples}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Inference", unit="sample")
            for future in pbar:
                sample = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    completed += 1
                except Exception as e:
                    tqdm.write(f"  Failed {sample.id}: {e}", file=sys.stderr)
                    failed += 1
                pbar.set_postfix(done=completed, failed=failed)

    print(f"\nDone. {completed} responses saved to {output_path}, {failed} failures.", file=sys.stderr)


if __name__ == "__main__":
    main()
