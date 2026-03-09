#!/usr/bin/env python3
"""Simple hyperparameter sweep using Optuna.

This file demonstrates how you could systematically explore genome hyperparameters
(like embedding dimension or gradient learning rate) by pretraining on a small
corpus and measuring perplexity on a validation split.  The study is not
intended to be production-grade; it's just a template you can extend with more
parameters and proper logging.

Usage:
    pip install optuna
    python run_experiment.py --dataset wikitext --split train --val_split test

"""

from __future__ import annotations

import argparse
import optuna
import torch

from pretrain import iter_sequences, tokenize
from model.model import Model
from model.genome import Genome, LayerTopology, HebbianParams, GenerationParams
from model.tensor import seed_all
from evaluate import evaluate_lm


def objective(trial: optuna.trial.Trial) -> float:
    # sample hyperparameters
    lr = trial.suggest_loguniform("gradient_lr", 1e-5, 1e-2)
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
    transformer_hidden = trial.suggest_int("transformer_hidden", 128, 1024, log=True)

    # build genome with these values
    topo = LayerTopology(embed_dim=embed_dim, transformer_hidden=transformer_hidden)
    genome = Genome(topology=topo)
    genome = genome.__class__(**{**genome.__dict__, "gradient": genome.gradient.__class__(lr=lr, weight_decay=0.0)})

    # quick pretrain on tiny number of sequences to get a encoder
    seqs = []
    for i, seq in enumerate(iter_sequences(cfg_dummy)):  # cfg_dummy defined later
        seqs.append(seq)
        if i >= 100:
            break

    model = Model(genome=genome, storage_dir="./tmp_expt", restore=False)
    model.birth(pretraining_corpus=seqs)
    model.session_start()

    # evaluate on a handful of validation texts
    val_texts = iter_texts_dummy  # defined later
    metrics = evaluate_lm(model, val_texts, max_seq_len=512, max_examples=200)
    return metrics["perplexity"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    # create iterators to re-use in objective closure
    global iter_texts_dummy
    def make_iter(split):
        from evaluate import iter_texts as _iter_texts
        return _iter_texts(args.dataset, split, seed=args.seed)
    iter_texts_dummy = make_iter(args.val_split)

    # dummy cfg for pretraining sequences
    from pretrain import TrainingConfig
    global cfg_dummy
    cfg_dummy = TrainingConfig(
        dataset=args.dataset,
        split=args.split,
        steps=200,
        streaming=False,
        max_seq_len=256,
        lr=0.01,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)

    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
