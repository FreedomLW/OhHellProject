import pytest

torch = pytest.importorskip("torch")

from rlohhell.policies.student_poker_policy import StudentPokerPolicy


def test_action_mask_zeroes_invalid_probabilities():
    policy = StudentPokerPolicy(
        public_state_dim=2,
        trick_embedding_dim=3,
        bid_action_size=4,
        play_action_size=5,
    )

    obs = {
        "hand": torch.zeros(1, 36),
        "public_state": torch.tensor([[1.0, 0.0]]),
        "trick": torch.zeros(1, 3),
    }
    action_mask = torch.tensor([[True, False, True, False]])

    logits, _ = policy(obs, phase="bid", action_mask=action_mask)
    probs = torch.softmax(logits, dim=-1)

    assert torch.all(probs[:, action_mask[0] == False] == 0)
