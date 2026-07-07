import pytest
torch = pytest.importorskip("torch")

from imml.fuse import (
    AttentionFusion,
    ConcatFusion,
    EmbraceNet,
    MaxFusion,
    MeanFusion,
    SumFusion,
)


@pytest.fixture
def sample_data():
    Xs = [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]),
    ]
    return Xs


def test_mean_fusion(sample_data):
    Xs = sample_data
    stacked_Xs = torch.stack(Xs)
    fusion = MeanFusion()
    out = fusion(Xs)
    assert out.shape == (2, 3)
    assert torch.allclose(out, stacked_Xs.mean(axis=0))


def test_sum_fusion(sample_data):
    Xs = sample_data
    stacked_Xs = torch.stack(Xs)
    fusion = SumFusion()
    out = fusion(Xs)
    assert out.shape == (2, 3)
    assert torch.allclose(out, stacked_Xs.sum(axis=0))


def test_max_fusion(sample_data):
    Xs = sample_data
    fusion = MaxFusion()
    out = fusion(Xs)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))


def test_concat_fusion(sample_data):
    Xs = sample_data
    fusion = ConcatFusion()
    out = fusion(Xs)
    assert out.shape == (2, 9)
    assert torch.allclose(
        out,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, -1.0, -2.0, -3.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0, -4.0, -5.0, -6.0],
            ]
        ),
    )


def test_attention_fusion_with_uniform_scores(sample_data):
    Xs = sample_data
    stacked_Xs = torch.stack(Xs)
    fusion = AttentionFusion(n_features=3, bias=False)
    with torch.no_grad():
        fusion.linear.weight.zero_()
        out = fusion(Xs)
    assert torch.allclose(out, torch.zeros_like(stacked_Xs[0]))


def test_attention_fusion_uses_softmax_scores(sample_data):
    Xs = sample_data
    Xs = [X[:, :2] for X in Xs[:2]]
    stacked_Xs = torch.stack(Xs)
    fusion = AttentionFusion(n_features=2, bias=False)
    with torch.no_grad():
        fusion.linear.weight.copy_(torch.eye(2))
        out = fusion(Xs)
    expected = torch.sum(stacked_Xs * torch.softmax(torch.tanh(stacked_Xs), dim=0), dim=0)

    assert torch.allclose(out, expected)


def test_embracenet_selection_probabilities_mark_missing_modalities(sample_data):
    Xs = [X[:, :2].clone() for X in sample_data[:2]]
    Xs[0][1] = 0.0
    stacked_Xs = torch.stack(Xs)
    fusion = EmbraceNet()

    p = fusion._get_selection_probabilities(stacked_Xs, b=2)

    assert torch.allclose(p[:, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(p[:, 1], torch.tensor([0.0, 1.0]))


def test_embracenet_sampling_indices_shape_and_one_hot(sample_data):
    Xs = [X[:, :2].clone() for X in sample_data[:2]]
    Xs[0][1] = 0.0
    stacked_Xs = torch.stack(Xs)
    fusion = EmbraceNet()
    p = fusion._get_selection_probabilities(stacked_Xs, b=2)

    torch.manual_seed(0)
    r = fusion._get_sampling_indices(p, c=4, m=2)

    assert r.shape == (2, 2, 4)
    assert torch.equal(r.sum(dim=0), torch.ones(2, 4, dtype=torch.long))
    assert torch.equal(r[0, 1], torch.zeros(4, dtype=torch.long))
    assert torch.equal(r[1, 1], torch.ones(4, dtype=torch.long))


def test_embracenet_forward_selects_only_available_modality(sample_data):
    Xs = [X.clone() for X in sample_data[:2]]
    Xs[0][0] = 0.0
    Xs[1][1] = 0.0
    stacked_Xs = torch.stack(Xs)
    fusion = EmbraceNet()

    torch.manual_seed(0)
    out = fusion(Xs)

    assert out.shape == (2, 3)
    assert torch.allclose(out[0], stacked_Xs[1, 0])
    assert torch.allclose(out[1], stacked_Xs[0, 1])


if __name__ == "__main__":
    pytest.main()
