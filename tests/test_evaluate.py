import torch
from model.model import Model
from model.genome import Genome
from evaluate import evaluate_lm
from model.tensor import seed_all


def test_evaluate_on_simple_sequence():
    seed_all(0)
    model = Model(genome=Genome(), restore=False)
    model.session_start()
    # artificially set a very short dataset of repeated tokens
    texts = ["hello world"]
    # patch tokenize to return a fixed sequence
    from pretrain import tokenize
    metrics = evaluate_lm(model, iter(texts), max_seq_len=32, max_examples=1)
    assert "loss" in metrics and "perplexity" in metrics
    assert metrics["loss"] >= 0.0
