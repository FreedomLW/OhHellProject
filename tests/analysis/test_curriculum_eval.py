import json

from rlohhell.analysis.curriculum_eval import load_oracle_scenarios, run_oracle_bootstrap


def test_load_oracle_scenarios_collects_unique_sorted_values(tmp_path):
    dataset = tmp_path / "oracle.jsonl"
    rows = [
        {"seed": 9, "round_size": 2},
        {"seed": 3, "round_size": 1},
        {"seed": 9, "round_size": 1},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    seeds, round_sizes = load_oracle_scenarios(str(dataset))

    assert seeds == [3, 9]
    assert round_sizes == [1, 2]


def test_oracle_bootstrap_is_deterministic_for_fixed_inputs():
    first = run_oracle_bootstrap(seeds=[5], round_sizes=[1, 2], iterations=2, rollouts_per_action=2)
    second = run_oracle_bootstrap(seeds=[5], round_sizes=[1, 2], iterations=2, rollouts_per_action=2)

    assert first.to_json() == second.to_json()
    assert len(first.iterations) == 2
    assert all(item.oracle_samples > 0 for item in first.iterations)
