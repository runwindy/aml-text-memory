from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.memory.dialogue_jepa import train_dialogue_jepa
from app.retrieval.embedding import build_embedding_provider


def iter_text_pairs(dataset_path: Path, max_cases: int | None = None):
    with dataset_path.open("r", encoding="utf-8-sig") as handle:
        for case_index, line in enumerate(handle):
            if max_cases is not None and case_index >= max_cases:
                break
            if not line.strip():
                continue
            case = json.loads(line)
            for session in case.get("sessions", []):
                history: list[str] = []
                for message in session.get("messages", []):
                    role = str(message.get("role", "user"))
                    content = str(message.get("content", "")).strip()
                    if not content:
                        continue
                    if role == "assistant" and history:
                        context = "\n".join(history[-6:])
                        yield context, content
                    history.append(f"[{role}] {content}")


async def run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    settings = Settings(
        embedding_provider=args.embedding_provider,
        embedding_dim=args.embedding_dim,
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_api_base,
        embedding_api_key=args.embedding_api_key,
        embedding_batch_size=args.embedding_batch_size,
        auth_mode="none",
        organizer_provider="rule",
        decomposer_provider="off",
        reranker_provider="lexical",
    )
    embedder = build_embedding_provider(settings)

    pairs = list(iter_text_pairs(dataset_path, max_cases=args.max_cases))
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise SystemExit("no (context, assistant) training pairs found")

    contexts = [context for context, _ in pairs]
    targets = [target for _, target in pairs]
    print(f"training_pairs={len(pairs)} embedding_provider={args.embedding_provider}", flush=True)

    context_vectors = await embedder.embed(contexts)
    target_vectors = await embedder.embed(targets)
    samples = list(zip(context_vectors, target_vectors))

    predictor = train_dialogue_jepa(
        samples,
        output_path=args.output,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(f"saved dialogue-jepa predictor: {args.output} input_dim={predictor.input_dim}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Dialogue-JEPA latent predictor.")
    parser.add_argument("--dataset", required=True, help="Converted LongMemEval/JSONL dataset")
    parser.add_argument("--output", default="data/dialogue_jepa.pt")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-provider", default="hashing", choices=["hashing", "openai"])
    parser.add_argument("--embedding-model", default="text-embedding-v4")
    parser.add_argument("--embedding-api-base", default="")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
