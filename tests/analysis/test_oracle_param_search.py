from rlohhell.analysis.curriculum_eval import find_optimal_oracle_params


def test_find_optimal_oracle_params_returns_candidate_from_grid():
    result = find_optimal_oracle_params(
        seeds=[3],
        round_sizes=[1],
        iterations_candidates=[1, 2],
        rollout_candidates=[1, 2],
    )

    assert result.iterations in {1, 2}
    assert result.rollouts_per_action in {1, 2}
    assert len(result.report.iterations) == result.iterations


def test_find_optimal_oracle_params_rejects_empty_candidates():
    try:
        find_optimal_oracle_params(seeds=[1], round_sizes=[1], iterations_candidates=[], rollout_candidates=[1])
        raise AssertionError("expected ValueError for empty iterations_candidates")
    except ValueError:
        pass

    try:
        find_optimal_oracle_params(seeds=[1], round_sizes=[1], iterations_candidates=[1], rollout_candidates=[])
        raise AssertionError("expected ValueError for empty rollout_candidates")
    except ValueError:
        pass
