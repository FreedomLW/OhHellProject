from rlohhell.analysis.oracle_dataset import OracleDatasetGenerator


def _sample_signature(samples):
    return [
        (
            sample.seed,
            sample.round_size,
            sample.phase,
            sample.seat,
            sample.chosen_action,
            tuple(sample.legal_actions),
            sample.action_value_margin,
        )
        for sample in samples
    ]


def test_oracle_dataset_is_reproducible_for_fixed_seeds():
    generator = OracleDatasetGenerator(rollouts_per_action=2, opponent_profile="random")

    run_a = generator.generate(seeds=[7], round_sizes=[1, 2], target_seat=0)
    run_b = generator.generate(seeds=[7], round_sizes=[1, 2], target_seat=0)

    assert _sample_signature(run_a) == _sample_signature(run_b)


def test_oracle_samples_include_required_metadata_and_legal_label():
    generator = OracleDatasetGenerator(rollouts_per_action=2, opponent_profile="heuristic")

    samples = generator.generate(seeds=[11], round_sizes=[1], target_seat=0)
    assert samples

    for sample in samples:
        assert sample.round_size == 1
        assert sample.phase in {"bid", "play"}
        assert sample.seat == 0
        assert sample.seed == 11
        assert sample.opponent_profile == "heuristic"
        assert sample.chosen_action in sample.legal_actions
