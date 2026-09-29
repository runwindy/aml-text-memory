from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence


def _lazy_torch():
    import torch  # type: ignore
    from torch import nn  # type: ignore

    return torch, nn


class DialogueJepaPredictor:
    """Small torch predictor for Dialogue-JEPA style latent expansion.

    The predictor maps context latents to target latents.  In this project it
    is used as an optional query expander: query vector -> predicted evidence
    vector -> blended query vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, device: str = "cpu") -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        torch, nn = _lazy_torch()
        self._torch = torch
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        ).to(device)

    def _normalize(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def predict(self, context_vector: Sequence[float]) -> list[float]:
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor([list(context_vector)], dtype=torch.float32, device=self.device)
            predicted = self.model(tensor)[0].detach().cpu().tolist()
        return self._normalize(predicted)

    def expand_query(
        self,
        query_vector: Sequence[float],
        *,
        blend_weight: float = 0.5,
    ) -> list[float]:
        query = self._normalize(query_vector)
        predicted = self.predict(query)
        weight = max(0.0, min(1.0, blend_weight))
        blended = [
            (1.0 - weight) * query_value + weight * predicted_value
            for query_value, predicted_value in zip(query, predicted)
        ]
        return self._normalize(blended)

    def save(self, path: str | Path) -> None:
        torch = self._torch
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.model.state_dict(),
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "DialogueJepaPredictor | None":
        model_path = Path(path)
        if not model_path.exists():
            return None
        torch, _ = _lazy_torch()
        payload = torch.load(model_path, map_location=device)
        predictor = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            device=device,
        )
        predictor.model.load_state_dict(payload["state_dict"])
        predictor.model.eval()
        return predictor


def train_dialogue_jepa(
    samples: Iterable[tuple[Sequence[float], Sequence[float]]],
    *,
    output_path: str | Path,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> DialogueJepaPredictor:
    """Train the latent predictor on (context, target) embedding pairs."""

    torch, _ = _lazy_torch()
    sample_list = [(list(context), list(target)) for context, target in samples]
    if not sample_list:
        raise ValueError("no training samples provided")
    input_dim = len(sample_list[0][0])
    if any(len(context) != input_dim or len(target) != input_dim for context, target in sample_list):
        raise ValueError("all context and target vectors must share one dimension")

    predictor = DialogueJepaPredictor(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    predictor.model.train()

    contexts = torch.tensor([context for context, _ in sample_list], dtype=torch.float32, device=device)
    targets = torch.tensor([target for _, target in sample_list], dtype=torch.float32, device=device)

    for _ in range(max(1, epochs)):
        permutation = torch.randperm(len(contexts), device=device)
        for start in range(0, len(contexts), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            predicted = predictor.model(contexts[indices])
            loss = loss_fn(predicted, targets[indices])
            loss.backward()
            optimizer.step()

    predictor.model.eval()
    predictor.save(output_path)
    return predictor

def _normalize_vector(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def pseudo_relevance_query_vector(
    query_vector: Sequence[float],
    evidence_vectors: Sequence[Sequence[float]],
    *,
    weight: float = 0.5,
) -> list[float]:
    """Training-free JEPA fallback.

    The centroid of the first-pass evidence vectors is used as a predicted
    evidence latent.  It is blended with the original query vector and used as
    an additional dense retrieval branch.
    """

    query = _normalize_vector(query_vector)
    valid = [
        _normalize_vector(vector)
        for vector in evidence_vectors
        if vector is not None and len(vector) == len(query)
    ]
    if not valid:
        return query
    dimension = len(query)
    centroid = [
        sum(vector[index] for vector in valid) / len(valid)
        for index in range(dimension)
    ]
    centroid = _normalize_vector(centroid)
    blend = max(0.0, min(1.0, weight))
    return _normalize_vector(
        [
            (1.0 - blend) * query[index] + blend * centroid[index]
            for index in range(dimension)
        ]
    )

