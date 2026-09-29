from __future__ import annotations

from app.memory.dialogue_jepa import (
    DialogueJepaPredictor,
    pseudo_relevance_query_vector,
    train_dialogue_jepa,
)


def test_dialogue_jepa_expand_query_keeps_dimension() -> None:
    predictor = DialogueJepaPredictor(input_dim=8, hidden_dim=16, device="cpu")
    expanded = predictor.expand_query([1.0] + [0.0] * 7, blend_weight=0.5)
    assert len(expanded) == 8
    assert any(value != 0 for value in expanded)


def test_dialogue_jepa_train_and_save(tmp_path) -> None:
    samples = [
        ([1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]),
        ([0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]),
        ([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1]),
        ([0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.9]),
    ]
    output = tmp_path / "dialogue_jepa.pt"
    predictor = train_dialogue_jepa(
        samples,
        output_path=output,
        hidden_dim=8,
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        device="cpu",
    )
    assert output.exists()

    loaded = DialogueJepaPredictor.load(output, device="cpu")
    assert loaded is not None
    assert len(loaded.predict([1.0, 0.0, 0.0, 0.0])) == 4

def test_pseudo_relevance_query_vector_blends_evidence() -> None:
    expanded = pseudo_relevance_query_vector(
        [1.0, 0.0, 0.0],
        [[0.0, 1.0, 0.0], [0.0, 0.8, 0.2]],
        weight=0.5,
    )
    assert len(expanded) == 3
    assert expanded[1] > 0

