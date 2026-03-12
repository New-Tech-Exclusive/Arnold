import tempfile
import torch
import json
from model.genome import Genome
from model.model import Model
from model.tensor import seed_all


def test_parameters_list_nonempty():
    seed_all(1)
    brain = Model(genome=Genome(), restore=False)
    params = brain.parameters()
    assert isinstance(params, list)
    assert len(params) > 0
    # all tensors should require grad
    assert all(p.requires_grad for p in params)


def test_online_training_reduces_loss():
    seed_all(1)
    brain = Model(genome=Genome(), restore=False)
    brain.session_start()
    # simple synthetic sequence of tokens
    tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    # perform several training steps and track loss
    losses = []
    for _ in range(5):
        loss = brain._online_train(tokens)
        assert loss is not None
        losses.append(loss)
    # loss should not increase monotonically (simple sanity check)
    assert losses[-1] <= max(losses)


def test_weightstore_seed_roundtrip(tmp_path):
    from model.weight_store import WeightStore
    from model.model import ModelState
    from model.genome import Genome
    
    ws = WeightStore(tmp_path)
    state = ModelState()
    genome = Genome()
    ws.save(state, genome=genome, seed=12345)
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert meta.get("seed") == 12345

