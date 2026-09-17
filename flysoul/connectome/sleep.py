"""Sleep: replaying the day's fights through the same dopamine machinery.

The learner that solved this boss in SoulsGym's own benchmark needed about five million
environment samples and reused each of them dozens of times from a replay buffer. This
circuit sees every sample once and throws it away, which is most of the gap between the
two, and it is not a gap biology leaves open either: flies consolidate mushroom body
memories during sleep, and the consolidation depends on the same dopaminergic circuitry
that wrote them.

So the fly sleeps between fights. Each step of the fight is recorded as it happens - the
spikes, the action the descending pools chose, the reinforcement that followed - and
while the game is reloading (dead time otherwise) the recording is played back through
`update_traces` and `apply_reinforcement`, several passes, over the last few dozen
fights. No rule about what to do is added anywhere: the circuit relives what happened
and the dopamine signal does exactly what it did the first time, only more often.

Replay is sequential within an episode because both the eligibility traces and the
temporal-difference critic depend on order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

OnStep = Callable[[int, int, int, "Transition", int, int], None]


@dataclass
class Transition:
    spikes: np.ndarray  # uint8 spike count per neuron for one game step
    channel: Optional[str]  # descending channel that was executed
    reward: float  # the learning signal that followed
    terminal: bool


@dataclass
class SleepReport:
    passes: int
    transitions: int
    episodes_in_memory: int
    duration_s: float
    weight_shift: float  # mean |dw| over plastic synapses, relative to their scale
    mean_abs_td: float


class SleepConsolidation:
    """Record fights while awake; replay them through plasticity while asleep."""

    def __init__(
        self,
        memory_episodes: int = 20,
        passes: int = 30,
        gain: float = 0.5,
        seed: int = 0,
        min_transitions: int = 4,
        max_transitions: int = 4000,
    ):
        self.memory_episodes = int(memory_episodes)
        self.passes = int(passes)
        # A night has to fit inside the game's loading screen. As the fly survives
        # longer the fights get longer, so the number of passes bends to a budget of
        # replayed steps rather than the other way round.
        self.max_transitions = int(max_transitions)
        # Each pass applies the learning rate again to the same experience. The gain
        # scales the rate during sleep so that thirty passes are not thirty times the
        # waking lesson; the replayed dopamine is a consolidation, not a new event.
        self.gain = float(gain)
        self.min_transitions = int(min_transitions)
        self.rng = np.random.default_rng(seed)
        self.memory: List[List[Transition]] = []
        self._current: List[Transition] = []
        self.last_report: Optional[SleepReport] = None

    # ------------------------------------------------------------------ awake

    def begin_episode(self) -> None:
        self._current = []

    def record(self, spike_counts: np.ndarray, channel: Optional[str], reward: float,
               terminal: bool) -> None:
        spikes = np.clip(np.asarray(spike_counts), 0, 255).astype(np.uint8)
        self._current.append(Transition(spikes, channel, float(reward), bool(terminal)))

    def end_episode(self) -> int:
        """File the fight just fought. Returns how many fights are in memory."""
        if len(self._current) >= self.min_transitions:
            self.memory.append(self._current)
            del self.memory[: max(0, len(self.memory) - self.memory_episodes)]
        self._current = []
        return len(self.memory)

    @property
    def transitions_in_memory(self) -> int:
        return sum(len(ep) for ep in self.memory)

    # ----------------------------------------------------------------- asleep

    def schedule(self) -> List[int]:
        """Which remembered fight each pass replays.

        The most recent fight always goes first - it is the one the waking brain has
        least consolidated - and the remaining passes draw uniformly from memory, so
        one unusual fight cannot dominate the night.
        """
        if not self.memory:
            return []
        latest = len(self.memory) - 1
        order = [latest]
        budget = self.max_transitions - len(self.memory[latest])
        for i in self.rng.integers(0, len(self.memory), size=max(0, self.passes - 1)):
            cost = len(self.memory[int(i)])
            if budget - cost < 0:
                break
            order.append(int(i))
            budget -= cost
        return order

    def consolidate(
        self,
        plasticity,
        on_step: Optional[OnStep] = None,
        target_duration_s: float = 0.0,
        stop: Optional[Callable[[], bool]] = None,
    ) -> SleepReport:
        """Replay memory through `plasticity`.

        Args:
            plasticity: The DopaminePlasticity whose traces and dopamine do the work.
            on_step: Called after every replayed transition with
                (pass_index, passes, memory_index, transition, done, total).
            target_duration_s: If positive, pace the replay to take about this long so a
                viewer can watch it. Zero replays as fast as it computes.
            stop: Optional predicate; when it returns True the night is cut short.
        """
        order = self.schedule()
        if not order:
            return SleepReport(0, 0, 0, 0.0, 0.0, 0.0)

        edges = plasticity.edges
        before = plasticity.topology.weight[edges].copy()
        lr0, critic0 = plasticity.lr, plasticity.critic_lr
        plasticity.lr = lr0 * self.gain
        plasticity.critic_lr = critic0 * self.gain

        total = sum(len(self.memory[i]) for i in order)
        pace = target_duration_s / total if target_duration_s > 0 and total else 0.0
        t0 = time.perf_counter()
        done = 0
        td_acc = 0.0
        try:
            for p, mem_index in enumerate(order):
                if stop is not None and stop():
                    break
                plasticity.reset()
                for tr in self.memory[mem_index]:
                    spikes = tr.spikes.astype(np.int32)
                    plasticity.update_traces(spikes, executed_channel=tr.channel)
                    plasticity.apply_reinforcement(tr.reward, spikes, terminal=tr.terminal)
                    done += 1
                    td_acc += abs(float(plasticity.last_td_error))
                    if on_step is not None:
                        on_step(p, len(order), mem_index, tr, done, total)
                    if pace > 0.0:
                        time.sleep(pace)
        finally:
            plasticity.lr = lr0
            plasticity.critic_lr = critic0
            plasticity.reset()

        after = plasticity.topology.weight[edges]
        shift = (
            float(np.mean(np.abs(after - before)) / plasticity._w_scale) if len(edges) else 0.0
        )
        report = SleepReport(
            passes=len(order),
            transitions=done,
            episodes_in_memory=len(self.memory),
            duration_s=time.perf_counter() - t0,
            weight_shift=shift,
            mean_abs_td=td_acc / max(1, done),
        )
        self.last_report = report
        return report
