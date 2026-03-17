"""Oracle bootstrapping pipeline for iterative policy teaching.

Pipeline:
1) sample one-round environments from fixed seeds/round sizes,
2) collect oracle labels versus a chosen opponent pool,
3) train a student policy to imitate oracle actions,
4) add the trained student to the opponent pool,
5) repeat.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import random
from typing import Callable, Dict, Iterable, List, Sequence

from rlohhell.analysis.oracle_dataset import OracleDatasetGenerator, OracleSample
from rlohhell.games.base import Card
from rlohhell.games.ohhell.game import OhHellGame
from rlohhell.games.ohhell.strategies import (
    BaseStrategy,
    ConservativeStrategy,
    GreedyStrategy,
    HeuristicStrategy,
    RandomStrategy,
)

StrategyFactory = Callable[[int], BaseStrategy]


def _default_pool() -> List[StrategyFactory]:
    return [
        lambda seed: _seeded(RandomStrategy(), seed),
        lambda seed: _seeded(GreedyStrategy(), seed),
        lambda seed: _seeded(HeuristicStrategy(), seed),
        lambda seed: _seeded(ConservativeStrategy(), seed),
    ]


@dataclass
class StageMetrics:
    stage: str
    avg_payoff: float
    win_rate: float
    episodes: int


@dataclass
class IterationResult:
    iteration: int
    oracle_samples: int
    teacher_pool_size: int
    against_defaults: StageMetrics
    against_teacher_pool: StageMetrics
    self_play: StageMetrics


@dataclass
class CurriculumReport:
    seeds: List[int]
    round_sizes: List[int]
    iterations: List[IterationResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class OracleParamSearchResult:
    iterations: int
    rollouts_per_action: int
    score: float
    win_rate: float
    report: CurriculumReport

    def to_json(self) -> str:
        return json.dumps(
            {
                "iterations": self.iterations,
                "rollouts_per_action": self.rollouts_per_action,
                "score": self.score,
                "win_rate": self.win_rate,
                "report": asdict(self.report),
            },
            indent=2,
        )


class ImitationStrategy(BaseStrategy):
    """Simple lookup-table student trained from oracle labels."""

    def __init__(self, policy_table: Dict[str, str] | None = None, fallback: BaseStrategy | None = None):
        self.policy_table = dict(policy_table or {})
        self.fallback = fallback or HeuristicStrategy()
        self.random_state = random.Random(0)

    def place_bid(self, hand, round_state, legal_actions=None):
        if legal_actions:
            return legal_actions[0]
        return 0

    def play_card(self, hand, trick_cards, legal_actions=None):
        if legal_actions:
            return legal_actions[0]
        return hand[0]

    def select_action(self, game, player_id):
        key = _state_key(game.get_state(player_id), player_id)
        action_key = self.policy_table.get(key)
        if action_key is not None:
            for action in game.get_legal_actions():
                if _action_to_str(action) == action_key:
                    return action
        return self.fallback.select_action(game, player_id)

    def clone(self, seed: int) -> "ImitationStrategy":
        cloned = ImitationStrategy(policy_table=self.policy_table, fallback=HeuristicStrategy())
        cloned.random_state = random.Random(seed)
        cloned.fallback.random_state = random.Random(seed + 41)
        return cloned


def train_imitation(samples: Sequence[OracleSample], fallback: BaseStrategy | None = None) -> ImitationStrategy:
    votes: Dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        key = _serialized_sample_key(sample.observation, sample.seat)
        votes[key][sample.chosen_action] += 1
    table = {key: counter.most_common(1)[0][0] for key, counter in votes.items()}
    return ImitationStrategy(policy_table=table, fallback=fallback)


def run_oracle_bootstrap(
    seeds: Sequence[int],
    round_sizes: Sequence[int],
    iterations: int = 3,
    rollouts_per_action: int = 4,
    target_seat: int = 0,
    hand_samples_per_seed: int = 1,
    play_phase_only: bool = True,
) -> CurriculumReport:
    teacher_pool = _default_pool()
    student = ImitationStrategy()
    reports: List[IterationResult] = []

    for idx in range(1, iterations + 1):
        generator = OracleDatasetGenerator(
            rollouts_per_action=rollouts_per_action,
            opponent_factory=_factory_for_pool(teacher_pool, target_seat=target_seat),
            target_rollout_strategy=HeuristicStrategy(),
            opponent_profile="curriculum_pool",
            hand_samples_per_seed=hand_samples_per_seed,
            play_phase_only=play_phase_only,
        )
        samples = generator.generate(seeds=seeds, round_sizes=round_sizes, target_seat=target_seat)
        student = train_imitation(samples, fallback=student)

        teacher_pool_with_student = teacher_pool + [lambda seed, s=student: s.clone(seed)]
        reports.append(
            IterationResult(
                iteration=idx,
                oracle_samples=len(samples),
                teacher_pool_size=len(teacher_pool_with_student),
                against_defaults=_evaluate(student, _default_pool(), seeds, round_sizes, target_seat),
                against_teacher_pool=_evaluate(student, teacher_pool_with_student, seeds, round_sizes, target_seat),
                self_play=_evaluate(student, [lambda seed, s=student: s.clone(seed)], seeds, round_sizes, target_seat),
            )
        )
        teacher_pool = teacher_pool_with_student

    return CurriculumReport(seeds=list(seeds), round_sizes=list(round_sizes), iterations=reports)


def find_optimal_oracle_params(
    seeds: Sequence[int],
    round_sizes: Sequence[int],
    iterations_candidates: Sequence[int],
    rollout_candidates: Sequence[int],
    target_seat: int = 0,
    hand_samples_per_seed: int = 1,
    play_phase_only: bool = True,
) -> OracleParamSearchResult:
    if not iterations_candidates:
        raise ValueError("iterations_candidates must not be empty")
    if not rollout_candidates:
        raise ValueError("rollout_candidates must not be empty")

    best: OracleParamSearchResult | None = None
    for num_iterations in sorted(set(int(v) for v in iterations_candidates)):
        for rollouts in sorted(set(int(v) for v in rollout_candidates)):
            report = run_oracle_bootstrap(
                seeds=seeds,
                round_sizes=round_sizes,
                iterations=num_iterations,
                rollouts_per_action=rollouts,
                target_seat=target_seat,
                hand_samples_per_seed=hand_samples_per_seed,
                play_phase_only=play_phase_only,
            )
            final_metrics = report.iterations[-1].against_teacher_pool
            candidate = OracleParamSearchResult(
                iterations=num_iterations,
                rollouts_per_action=rollouts,
                score=final_metrics.avg_payoff,
                win_rate=final_metrics.win_rate,
                report=report,
            )
            if best is None:
                best = candidate
                continue

            if candidate.score > best.score:
                best = candidate
            elif candidate.score == best.score and candidate.win_rate > best.win_rate:
                best = candidate
            elif (
                candidate.score == best.score
                and candidate.win_rate == best.win_rate
                and (candidate.iterations, candidate.rollouts_per_action) < (best.iterations, best.rollouts_per_action)
            ):
                best = candidate

    assert best is not None
    return best


def load_oracle_scenarios(dataset_jsonl: str) -> tuple[List[int], List[int]]:
    seeds: set[int] = set()
    round_sizes: set[int] = set()
    with open(dataset_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            seeds.add(int(row["seed"]))
            round_sizes.add(int(row["round_size"]))
    return sorted(seeds), sorted(round_sizes)


def _evaluate(student: BaseStrategy, opponent_pool: Sequence[StrategyFactory], seeds: Sequence[int], round_sizes: Sequence[int], target_seat: int) -> StageMetrics:
    payoffs: List[float] = []
    wins = 0
    episodes = 0
    for seed in seeds:
        for round_size in round_sizes:
            game = _new_game(seed=seed, round_size=round_size, num_players=4)
            while not game.is_over():
                pid = game.get_player_id()
                if pid == target_seat:
                    action = student.select_action(game, pid)
                else:
                    factory = opponent_pool[(seed + round_size + pid) % len(opponent_pool)]
                    opp = factory(seed + pid * 997 + round_size)
                    action = opp.select_action(game, pid)
                game.step(action)
            scores = game.get_payoffs()
            payoff = float(scores[target_seat])
            payoffs.append(payoff)
            if payoff >= max(scores):
                wins += 1
            episodes += 1
    avg = sum(payoffs) / len(payoffs) if payoffs else 0.0
    stage = "self_play" if len(opponent_pool) == 1 else "evaluation"
    return StageMetrics(stage=stage, avg_payoff=avg, win_rate=wins / episodes if episodes else 0.0, episodes=episodes)


def _factory_for_pool(pool: Sequence[StrategyFactory], target_seat: int) -> Callable[[int], List[BaseStrategy]]:
    def _factory(seed: int) -> List[BaseStrategy]:
        out: List[BaseStrategy] = []
        for seat in range(4):
            if seat == target_seat:
                out.append(_seeded(RandomStrategy(), seed + seat))
            else:
                factory = pool[(seed + seat) % len(pool)]
                out.append(factory(seed + seat * 101))
        return out

    return _factory


def _new_game(seed: int, round_size: int, num_players: int) -> OhHellGame:
    import numpy as np

    game = OhHellGame(num_players=num_players)
    game.np_random = np.random.RandomState(seed)
    game.current_player = random.Random(seed).randint(0, num_players - 1)
    game.init_game()
    game.round_sequence = [round_size]
    game.max_rounds = 1
    game.current_round = 0
    _, pid = game._start_round(round_size, keep_scores=False)
    game.current_player = pid
    return game


def _seeded(strategy: BaseStrategy, seed: int) -> BaseStrategy:
    strategy.random_state = random.Random(seed)
    return strategy


def _action_to_str(action: object) -> str:
    if isinstance(action, Card):
        mode = getattr(action, "joker_mode", None)
        return f"{action.get_index()}:{mode}" if mode else action.get_index()
    return str(action)


def _serialized_sample_key(observation: Dict[str, object], seat: int) -> str:
    return json.dumps({"seat": seat, "obs": observation}, sort_keys=True)


def _state_key(state: Dict[str, object], seat: int) -> str:
    compact = {
        "hand": [c.get_index() for c in state.get("hand", [])],
        "played_cards": [c.get_index() for c in state.get("played_cards", [])],
        "proposed_tricks": int(state.get("proposed_tricks", 0)),
        "tricks_won": int(state.get("tricks_won", 0)),
        "players_tricks_won": [int(v) for v in state.get("players_tricks_won", [])],
        "current_player": int(state.get("current_player", 0)),
        "trump_card": str(state.get("trump_card")),
    }
    return _serialized_sample_key(compact, seat)
