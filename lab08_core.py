"""Lab 08: the shop and harness of Lab 07, plus a way to save and restore the shop.

This is ``lab07_core.py`` with ONE addition: ``InventoryEnv.get_state`` and
``InventoryEnv.set_state``, so that a program can stop, write the shop to a file and continue
later with the same demand stream. Nothing else differs. Do not change this file.

Everything down to ``exact_value`` is the Lecture 11.1 shop (``inventory_mdp.py``) with ONE
addition: ``ShopParams.order_fixed_cost``, a cost paid on every day on which anything is ordered.
It defaults to 0, and with 0 every function below returns exactly what ``inventory_mdp.py``
returns (a test checks this). The comparison harness (value iteration driver, Q-learning loop,
the scoring rule, ``compare_strategies``) comes after ``exact_value``, under its own heading.

A one-product shop as a Markov decision process: every morning the manager sees how many units
are on the shelf (the *state*), decides how many units to order (the *action*), the order
arrives at once, demand for the day is drawn from a fixed distribution, and the day ends with a
profit or a loss in tenge (the *reward*).

The module needs numpy only. The interface has the shape of a Gymnasium environment (``reset``
returns ``(observation, info)``, ``step`` returns
``(observation, reward, terminated, truncated, info)``) but does not import Gymnasium.

Design choices that matter:

* The prices, costs and the demand distribution are *synthetic*. They were picked to give a
  readable picture (a visible best order-up-to level, a visible loss on both sides of it), not
  calibrated to any real shop.
* Lead time is zero: an order arrives before the day's demand. With a delivery delay the stock
  on the shelf alone would stop being a Markov state (the units on the road would matter too).
* Unmet demand is lost, not backlogged.
* There is no terminal state. An episode ends only because the calendar says so, which makes it
  *truncated*, never *terminated*, in Gymnasium's terms.
* Demand is drawn by inverting the cumulative distribution with ``rng.random()``, so a seed
  gives the same demand sequence on every numpy version that keeps that stream.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SeedLike = int | np.random.SeedSequence | None


@dataclass(frozen=True)
class ShopParams:
    """Everything that defines the shop. Money is in tenge, quantities in units.

    Attributes:
        capacity: Largest stock the shelf holds. States and actions are ``0..capacity``.
        price: Selling price of one unit.
        unit_cost: Purchase cost of one unit, paid when the order is placed.
        holding_cost: Cost of one unit left on the shelf at the end of the day.
        shortage_cost: Goodwill cost of one unit of unmet demand, on top of the lost margin.
        demand_probs: ``demand_probs[d]`` is the probability that ``d`` units are demanded.
        start_stock: Stock at the start of an episode.
        horizon: Number of days in an episode.
        order_fixed_cost: Cost paid on a day when ``order > 0``, whatever the size of the order
            (a delivery fee). Lab 07 addition; 0 reproduces the Lecture 11.1 shop exactly.
    """

    capacity: int = 10
    price: int = 500
    unit_cost: int = 300
    holding_cost: int = 40
    shortage_cost: int = 100
    demand_probs: tuple[float, ...] = (0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05)
    start_stock: int = 5
    horizon: int = 100
    order_fixed_cost: int = 0

    def __post_init__(self) -> None:
        """Reject parameter sets that do not define a proper shop."""
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if not 0 <= self.start_stock <= self.capacity:
            raise ValueError("start_stock must be between 0 and capacity")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if min(self.price, self.unit_cost, self.holding_cost, self.shortage_cost,
               self.order_fixed_cost) < 0:
            raise ValueError("prices and costs must not be negative")
        probs = np.asarray(self.demand_probs, dtype=float)
        if probs.ndim != 1 or probs.size == 0 or (probs < 0).any():
            raise ValueError("demand_probs must be a non-empty vector of non-negative numbers")
        if abs(probs.sum() - 1.0) > 1e-9:
            raise ValueError(f"demand_probs must sum to 1, got {probs.sum()!r}")

    @property
    def n_states(self) -> int:
        """Number of stock levels, ``capacity + 1``."""
        return self.capacity + 1

    @property
    def n_actions(self) -> int:
        """Number of order sizes, ``capacity + 1`` (only ``capacity - stock + 1`` are legal)."""
        return self.capacity + 1

    @property
    def demand_cdf(self) -> np.ndarray:
        """Cumulative demand distribution, last entry forced to exactly 1.0."""
        cdf = np.cumsum(np.asarray(self.demand_probs, dtype=float))
        cdf[-1] = 1.0
        return cdf

    @property
    def mean_demand(self) -> float:
        """Expected units demanded per day."""
        return float(np.dot(np.arange(len(self.demand_probs)), self.demand_probs))

    def legal_orders(self, stock: int) -> range:
        """Order sizes that fit on the shelf when ``stock`` units are already there."""
        return range(self.capacity - stock + 1)


@dataclass(frozen=True)
class DayOutcome:
    """The accounts of one day. ``reward`` is what the agent receives."""

    stock: int
    ordered: int
    demand: int
    sales: int
    shortfall: int
    end_stock: int
    revenue: int
    purchase: int
    holding: int
    penalty: int
    reward: int


def day_outcome(params: ShopParams, stock: int, order: int, demand: int) -> DayOutcome:
    """Book one day. The only place where the reward is defined.

    Args:
        params: The shop.
        stock: Units on the shelf in the morning.
        order: Units ordered; they arrive before demand is served.
        demand: Units customers ask for.

    Returns:
        The day's quantities and the reward ``revenue - purchase - holding - penalty``.
    """
    available = stock + order
    sales = min(available, demand)
    shortfall = demand - sales
    end_stock = available - sales
    revenue = params.price * sales
    purchase = params.unit_cost * order + (params.order_fixed_cost if order > 0 else 0)
    holding = params.holding_cost * end_stock
    penalty = params.shortage_cost * shortfall
    return DayOutcome(stock, order, demand, sales, shortfall, end_stock, revenue, purchase,
                      holding, penalty, revenue - purchase - holding - penalty)


class InventoryEnv:
    """The shop as an environment with a Gymnasium-shaped interface.

    The observation is the stock on the shelf (an ``int``); the action is the order size.
    """

    def __init__(self, params: ShopParams | None = None) -> None:
        """Create the environment; call ``reset`` before the first ``step``."""
        self.params = params or ShopParams()
        self._cdf = self.params.demand_cdf
        self._rng: np.random.Generator | None = None
        self._stock: int | None = None
        self._day = 0

    def reset(self, *, seed: SeedLike = None,
              options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Seeds the demand stream. ``None`` keeps the current stream (as Gymnasium does),
                or creates an unseeded one on the first call.
            options: ``{"stock": n}`` overrides the starting stock.

        Returns:
            ``(observation, info)``: the starting stock and an empty dict.
        """
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        stock = self.params.start_stock
        if options and "stock" in options:
            stock = operator.index(options["stock"])
            if not 0 <= stock <= self.params.capacity:
                raise ValueError("options['stock'] must be between 0 and capacity")
        self._stock, self._day = stock, 0
        return stock, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Order ``action`` units, let one day of demand pass, book the day.

        Args:
            action: Order size, an integer in ``0..capacity - stock``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``. ``terminated`` is always
            ``False`` (the shop has no end state); ``truncated`` becomes ``True`` after the
            ``horizon``-th day. ``info`` holds the day's accounts (see ``DayOutcome``).

        Raises:
            RuntimeError: Before ``reset`` or after the episode has been truncated.
            TypeError: If ``action`` is not an integer.
            ValueError: If the order does not fit on the shelf.
        """
        if self._stock is None or self._rng is None:
            raise RuntimeError("call reset() before step()")
        if self._day >= self.params.horizon:
            raise RuntimeError("the episode is over; call reset()")
        order = operator.index(action)
        if order not in self.params.legal_orders(self._stock):
            raise ValueError(
                f"order {order} does not fit: stock {self._stock}, capacity {self.params.capacity}")
        demand = int(np.searchsorted(self._cdf, self._rng.random(), side="right"))
        outcome = day_outcome(self.params, self._stock, order, demand)
        self._stock = outcome.end_stock
        self._day += 1
        truncated = self._day >= self.params.horizon
        info = asdict(outcome)
        info["day"] = self._day
        return outcome.end_stock, float(outcome.reward), False, truncated, info

    def get_state(self) -> dict[str, Any]:
        """Everything the next ``step`` depends on, as plain values that ``json`` can write.

        Returns:
            ``{"stock": int | None, "day": int, "rng": dict | None}``. ``day`` is the number of
            days already played in the current episode; ``rng`` is the state of the demand
            generator. ``stock`` and ``rng`` are ``None`` before the first ``reset``.
        """
        return {
            "stock": self._stock,
            "day": self._day,
            "rng": None if self._rng is None else self._rng.bit_generator.state,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore what ``get_state`` returned (a round trip through ``json`` is fine).

        Args:
            state: A value of ``get_state`` of a shop with the same parameters.

        Raises:
            ValueError: If ``state`` is not something ``get_state`` could have returned. The
                environment is left unchanged then.
        """
        try:
            stock, day, rng_state = state["stock"], operator.index(state["day"]), state["rng"]
            if stock is not None:
                stock = operator.index(stock)
                if not 0 <= stock <= self.params.capacity:
                    raise ValueError(f"stock {stock} is outside 0..{self.params.capacity}")
            if not 0 <= day <= self.params.horizon:
                raise ValueError(f"day {day} is outside 0..{self.params.horizon}")
            rng = None
            if rng_state is not None:
                rng = np.random.default_rng(0)
                rng.bit_generator.state = rng_state
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"not a saved InventoryEnv state: {exc!r}") from exc
        self._rng, self._stock, self._day = rng, stock, day


class TablePolicy:
    """A policy as a table ``probs[state, action]`` (each row a distribution over legal orders)."""

    def __init__(self, probs: np.ndarray, name: str = "") -> None:
        """Validate the table.

        Args:
            probs: Array of shape ``(capacity + 1, capacity + 1)``.
            name: Label used in printouts.

        Raises:
            ValueError: If rows are not distributions, or put weight on an order that does not
                fit on the shelf.
        """
        table = np.asarray(probs, dtype=float)
        if table.ndim != 2 or table.shape[0] != table.shape[1]:
            raise ValueError("probs must be a square (states x actions) table")
        if (table < 0).any() or not np.allclose(table.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("every row of probs must be a probability distribution")
        capacity = table.shape[0] - 1
        for stock in range(table.shape[0]):
            if table[stock, capacity - stock + 1:].any():
                raise ValueError(f"state {stock}: probability on an order that does not fit")
        self.probs = table
        self.name = name
        cum = np.cumsum(table, axis=1)
        cum[:, -1] = 1.0
        self._cum = cum

    def act(self, state: int, rng: np.random.Generator) -> int:
        """Draw an order for ``state`` (one ``rng.random()`` call, deterministic or not)."""
        return int(np.searchsorted(self._cum[state], rng.random(), side="right"))


def uniform_random_policy(params: ShopParams) -> TablePolicy:
    """Order a uniformly random number of units among those that fit."""
    probs = np.zeros((params.n_states, params.n_actions))
    for stock in range(params.n_states):
        probs[stock, : params.capacity - stock + 1] = 1.0 / (params.capacity - stock + 1)
    return TablePolicy(probs, "random")


def never_order_policy(params: ShopParams) -> TablePolicy:
    """Never order anything."""
    probs = np.zeros((params.n_states, params.n_actions))
    probs[:, 0] = 1.0
    return TablePolicy(probs, "never order")


def order_up_to_policy(params: ShopParams, level: int) -> TablePolicy:
    """Every morning order enough to bring the shelf up to ``level`` units (nothing if above)."""
    if not 0 <= level <= params.capacity:
        raise ValueError("level must be between 0 and capacity")
    probs = np.zeros((params.n_states, params.n_actions))
    for stock in range(params.n_states):
        probs[stock, max(level - stock, 0)] = 1.0
    return TablePolicy(probs, f"order up to {level}")


def policy_from_function(params: ShopParams, rule: Callable[[int], int],
                         name: str = "custom") -> TablePolicy:
    """Turn a deterministic rule ``stock -> order`` into a policy table.

    Raises:
        ValueError: If the rule returns an order that does not fit for some stock level.
    """
    probs = np.zeros((params.n_states, params.n_actions))
    for stock in range(params.n_states):
        order = operator.index(rule(stock))
        if order not in params.legal_orders(stock):
            raise ValueError(f"rule({stock}) = {order} does not fit on the shelf")
        probs[stock, order] = 1.0
    return TablePolicy(probs, name)


@dataclass(frozen=True)
class Episode:
    """One simulated episode. ``steps`` holds ``(state, action, reward, next_state)`` tuples."""

    steps: tuple[tuple[int, int, float, int], ...]
    discounted_return: float
    undiscounted_return: float


def episode_seeds(seed: int, n: int) -> list[tuple[np.random.SeedSequence, np.random.SeedSequence]]:
    """``n`` independent (environment, policy) seed pairs derived from one integer."""
    pairs = []
    for child in np.random.SeedSequence(seed).spawn(n):
        env_seed, policy_seed = child.spawn(2)
        pairs.append((env_seed, policy_seed))
    return pairs


def rollout(params: ShopParams, policy: TablePolicy, gamma: float,
            env_seed: SeedLike, policy_seed: SeedLike = None,
            start_stock: int | None = None) -> Episode:
    """Run one episode and collect the return ``G = sum_k gamma**k * R_{k+1}``.

    Args:
        params: The shop.
        policy: The policy to follow.
        gamma: Discount factor in ``[0, 1]``.
        env_seed: Seed of the demand stream.
        policy_seed: Seed of the policy's own randomness.
        start_stock: Overrides ``params.start_stock``.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    env = InventoryEnv(params)
    options = None if start_stock is None else {"stock": start_stock}
    state, _ = env.reset(seed=env_seed, options=options)
    policy_rng = np.random.default_rng(policy_seed)
    steps: list[tuple[int, int, float, int]] = []
    discounted = undiscounted = 0.0
    weight = 1.0
    done = False
    while not done:
        action = policy.act(state, policy_rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps.append((state, action, reward, next_state))
        discounted += weight * reward
        undiscounted += reward
        weight *= gamma
        state, done = next_state, terminated or truncated
    return Episode(tuple(steps), discounted, undiscounted)


def mc_return(params: ShopParams, policy: TablePolicy, gamma: float, n_episodes: int,
              seed: int, start_stock: int | None = None) -> tuple[float, float]:
    """Monte Carlo estimate of the expected discounted return and its standard error.

    Episode ``i`` uses the same seeds for every policy, so two policies are compared on the
    same demand sequences.
    """
    if n_episodes < 2:
        raise ValueError("need at least two episodes for a standard error")
    returns = np.array([
        rollout(params, policy, gamma, env_seed, policy_seed, start_stock).discounted_return
        for env_seed, policy_seed in episode_seeds(seed, n_episodes)
    ])
    return float(returns.mean()), float(returns.std(ddof=1) / np.sqrt(n_episodes))


def sweep_order_up_to(params: ShopParams, gamma: float, n_episodes: int,
                      seed: int) -> list[tuple[int, float, float]]:
    """``(level, mean return, standard error)`` for every order-up-to level ``0..capacity``."""
    return [(level, *mc_return(params, order_up_to_policy(params, level), gamma, n_episodes, seed))
            for level in range(params.capacity + 1)]


def transition_model(params: ShopParams) -> tuple[np.ndarray, np.ndarray]:
    """The exact dynamics of the shop.

    Returns:
        ``P`` of shape ``(actions, states, states)``, ``P[a, s, s2] = Pr(next stock s2 | s, a)``,
        and ``R`` of shape ``(states, actions)``, the expected reward of ordering ``a`` at stock
        ``s``. Rows of orders that do not fit are all zero.
    """
    n = params.n_states
    p_next = np.zeros((params.n_actions, n, n))
    reward = np.zeros((n, params.n_actions))
    for stock in range(n):
        for order in params.legal_orders(stock):
            for demand, prob in enumerate(params.demand_probs):
                outcome = day_outcome(params, stock, order, demand)
                p_next[order, stock, outcome.end_stock] += prob
                reward[stock, order] += prob * outcome.reward
    return p_next, reward


def exact_value(params: ShopParams, policy: TablePolicy, gamma: float,
                horizon: int | None = None) -> np.ndarray:
    """Expected discounted return from each starting stock, computed exactly.

    Args:
        params: The shop.
        policy: The policy to evaluate.
        gamma: Discount factor.
        horizon: Number of days. ``None`` means an endless shop (needs ``gamma < 1``).

    Returns:
        A vector indexed by the starting stock.
    """
    p_next, reward = transition_model(params)
    p_pi = np.einsum("sa,ast->st", policy.probs, p_next)
    r_pi = np.einsum("sa,sa->s", policy.probs, reward)
    if horizon is None:
        if gamma >= 1.0:
            raise ValueError("an endless shop needs gamma < 1")
        return np.linalg.solve(np.eye(params.n_states) - gamma * p_pi, r_pi)
    value = np.zeros(params.n_states)
    for _ in range(horizon):
        value = r_pi + gamma * p_pi @ value
    return value


# =============================================================================================
# Lab 07 comparison harness (unchanged in Lab 08). Three routes to a good ordering policy, scored by ONE rule.
#
#   1. order-up-to sweep : simulation only (no model, no learning)
#   2. value iteration   : uses the transition table P and R (a model)
#   3. Q-learning        : uses samples from the shop only (no model)
#
# The scoring rule, the same for every route: the exact expected discounted return of the
# policy the route produced, from the starting stock, in an endless shop (``exact_value`` with
# ``horizon=None``, gamma < 1). Value iteration solves that same endless-shop problem, so its
# score is the optimum and no policy can score above it (up to rounding).
# =============================================================================================


def legal_mask(params: ShopParams) -> np.ndarray:
    """Boolean array ``(states, actions)``, True where the order fits on the shelf."""
    mask = np.zeros((params.n_states, params.n_actions), dtype=bool)
    for stock in range(params.n_states):
        mask[stock, : params.capacity - stock + 1] = True
    return mask


def greedy_policy(q: np.ndarray, mask: np.ndarray, name: str = "greedy") -> TablePolicy:
    """The deterministic policy that takes the best legal action of ``q`` in every state.

    Ties go to the smaller order. Illegal actions are never chosen, whatever ``q`` holds there.
    """
    masked = np.where(mask, q, -np.inf)
    probs = np.zeros(q.shape)
    probs[np.arange(q.shape[0]), masked.argmax(axis=1)] = 1.0
    return TablePolicy(probs, name)


def score_policy(params: ShopParams, policy: TablePolicy, gamma: float) -> float:
    """The scoring rule of the lab: exact expected discounted return of ``policy`` from the
    starting stock, in an endless shop. Needs ``gamma < 1``."""
    return float(exact_value(params, policy, gamma, horizon=None)[params.start_stock])


@dataclass(frozen=True)
class StrategyResult:
    """What one run of one route produced, and what it cost.

    Attributes:
        name: The route.
        policy: The policy it produced.
        score: ``score_policy`` of that policy (tenge).
        env_steps: Days drawn from the shop (samples). 0 for value iteration.
        backups: Updates of one (state, action) pair. Value iteration: sweeps x legal pairs
            (each backup is an *expected* update over the whole table). Q-learning: one per day
            (each backup is a *sample* update). The sweep route does no backups.
        seconds: Wall-clock time of the run (this machine, one run; indicative only).
        detail: A short note (the chosen level, the number of sweeps, ...).
    """

    name: str
    policy: TablePolicy
    score: float
    env_steps: int
    backups: int
    seconds: float
    detail: str = ""


BackupFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
ChooseFn = Callable[[np.ndarray, int, float, np.ndarray, np.random.Generator], int]
UpdateFn = Callable[[np.ndarray, int, int, float, int, float, float, np.ndarray], None]


def run_value_iteration(params: ShopParams, gamma: float, backup: BackupFn,
                        tol: float = 1e-6, max_sweeps: int = 10_000) -> StrategyResult:
    """Value iteration driven by YOUR ``backup``.

    Each sweep calls ``backup(p_next, reward, mask, v, gamma)``, takes ``v_new = q.max(axis=1)``
    and stops when no state's value moved by more than ``tol`` tenge.

    Args:
        params: The shop.
        gamma: Discount factor, below 1.
        backup: Returns ``q`` of shape ``(states, actions)`` from the current values ``v``.
        tol: Stopping tolerance in tenge.
        max_sweeps: Safety limit.

    Raises:
        RuntimeError: If the values have not settled after ``max_sweeps`` sweeps.
        ValueError: If ``backup`` returns an array of the wrong shape or with non-finite
            values on a legal action.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("value iteration here needs 0 <= gamma < 1")
    started = time.perf_counter()
    p_next, reward = transition_model(params)
    mask = legal_mask(params)
    v = np.zeros(params.n_states)
    q = np.zeros(mask.shape)
    for sweep in range(1, max_sweeps + 1):
        q = np.asarray(backup(p_next, reward, mask, v, gamma), dtype=float)
        if q.shape != mask.shape:
            raise ValueError(f"backup returned shape {q.shape}, expected {mask.shape}")
        if not np.isfinite(q[mask]).all():
            raise ValueError("backup returned a non-finite value on a legal action")
        v_new = np.where(mask, q, -np.inf).max(axis=1)
        moved = float(np.abs(v_new - v).max())
        v = v_new
        if moved < tol:
            policy = greedy_policy(q, mask, "value iteration")
            return StrategyResult("value iteration", policy, score_policy(params, policy, gamma),
                                  env_steps=0, backups=sweep * int(mask.sum()),
                                  seconds=time.perf_counter() - started,
                                  detail=f"{sweep} sweeps")
    raise RuntimeError(f"values still moving by {moved:.3g} after {max_sweeps} sweeps")


def run_q_learning(params: ShopParams, gamma: float, n_steps: int, seed: int,
                   choose_action: ChooseFn, update: UpdateFn,
                   alpha: float = 0.5, alpha_half_life: float = 50.0,
                   epsilon: float = 0.2) -> StrategyResult:
    """Tabular Q-learning driven by YOUR ``choose_action`` and ``update``.

    The agent lives in the shop for ``n_steps`` days. When the 100-day episode is cut off by the
    calendar the environment is reset, and the update of the last day still bootstraps from the
    stock that day ended with: the cut is *truncation*, not *termination* (the shop has no end
    state), so the future is not worth zero. The agent never sees ``P`` or ``R``.

    Args:
        params: The shop (used only to build the environment and the legality mask).
        gamma: Discount factor, below 1 (the scoring rule needs it).
        n_steps: Days of experience (samples) the agent may use.
        seed: Seeds the demand stream and the agent's own exploration independently.
        choose_action: ``(q, state, epsilon, mask, rng) -> order``, epsilon-greedy.
        update: ``(q, state, action, reward, next_state, alpha, gamma, mask) -> None``, changes
            ``q[state, action]`` in place.
        alpha: Step size of a pair's first update.
        alpha_half_life: The step size of the ``n``-th update of one (state, action) pair is
            ``alpha / (1 + n / alpha_half_life)``: it halves after that many visits of the pair.
            ``float("inf")`` keeps the step size constant. With a constant step size the final
            greedy policy keeps jittering around the answer however many days it has seen, so
            more days need not mean a better policy; a decaying step size does not have that
            problem (the README gives the measurement).
        epsilon: Probability of a random legal order.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("Q-learning here needs 0 <= gamma < 1")
    if alpha_half_life <= 0:
        raise ValueError("alpha_half_life must be positive (use float('inf') for a constant step)")
    started = time.perf_counter()
    env_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
    agent_rng = np.random.default_rng(agent_seed)
    env = InventoryEnv(params)
    mask = legal_mask(params)
    q = np.zeros(mask.shape)
    q[~mask] = -np.inf
    visits = np.zeros(mask.shape)
    state, _ = env.reset(seed=env_seed)
    for _ in range(n_steps):
        action = int(choose_action(q, state, epsilon, mask, agent_rng))
        next_state, reward, terminated, truncated, _ = env.step(action)
        step = alpha / (1.0 + visits[state, action] / alpha_half_life)
        visits[state, action] += 1
        update(q, state, action, reward, next_state, step, gamma, mask)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()  # seed=None keeps the demand stream going
    policy = greedy_policy(q, mask, "Q-learning")
    return StrategyResult("Q-learning", policy, score_policy(params, policy, gamma),
                          env_steps=n_steps, backups=n_steps,
                          seconds=time.perf_counter() - started,
                          detail=f"alpha={alpha}, half-life={alpha_half_life}, epsilon={epsilon}")


def run_order_up_to_sweep(params: ShopParams, gamma: float, n_episodes: int,
                          seed: int) -> StrategyResult:
    """Try every order-up-to level for ``n_episodes`` simulated episodes, keep the best mean.

    The route sees noisy simulated returns only, never the table. Its pick can be wrong when two
    levels are close, and that is part of what the comparison shows.
    """
    started = time.perf_counter()
    rows = sweep_order_up_to(params, gamma, n_episodes, seed)
    level = max(rows, key=lambda row: row[1])[0]
    policy = order_up_to_policy(params, level)
    return StrategyResult("order-up-to sweep", policy, score_policy(params, policy, gamma),
                          env_steps=len(rows) * n_episodes * params.horizon, backups=0,
                          seconds=time.perf_counter() - started,
                          detail=f"picked level {level}")


def describe_policy(policy: TablePolicy) -> str:
    """A deterministic policy in words: ``"order up to S when stock <= s"`` if it has that form.

    Returns ``"never orders"``, ``"irregular"`` (not of that form, or randomised) or the
    ``(s, S)`` rule, e.g. ``"order up to 6 when stock <= 2"``. ``S = s + 1`` is the
    order-up-to-S policy of Lecture 11.1.
    """
    probs = policy.probs
    if not np.allclose(probs.max(axis=1), 1.0):
        return "irregular"
    orders = probs.argmax(axis=1)
    ordering = [stock for stock in range(len(orders)) if orders[stock] > 0]
    if not ordering:
        return "never orders"
    reorder, top = max(ordering), int(orders[0])
    expected = [max(top - stock, 0) if stock <= reorder else 0 for stock in range(len(orders))]
    if orders.tolist() != expected:
        return "irregular"
    return f"order up to {top} when stock <= {reorder}"


@dataclass(frozen=True)
class Comparison:
    """The three routes on one shop, with the stochastic routes repeated over ``seeds``.

    Attributes:
        params, gamma: The shop and the discount factor all three routes shared.
        vi: Value iteration (deterministic: one run).
        sweep: The sweep route, one result per seed.
        q: Q-learning, one result per seed.
    """

    params: ShopParams
    gamma: float
    vi: StrategyResult
    sweep: tuple[StrategyResult, ...]
    q: tuple[StrategyResult, ...]

    @property
    def optimum(self) -> float:
        """The score of value iteration, the best any policy can score."""
        return self.vi.score

    def regret(self, result: StrategyResult) -> float:
        """Tenge per start-stock value lost against the optimum (never meaningfully negative)."""
        return self.optimum - result.score

    def regret_of(self, runs: tuple[StrategyResult, ...]) -> float:
        """Mean regret (tenge) of several runs of one route, e.g. ``cmp.regret_of(cmp.sweep)``."""
        return float(self.optimum - np.mean([r.score for r in runs]))

    def rows(self) -> list[dict[str, float | int | str]]:
        """One dict per route: the numbers ``table`` prints, as data (for ``results.json``)."""
        out: list[dict[str, float | int | str]] = []
        for name, runs in (("order-up-to sweep", self.sweep), ("value iteration", (self.vi,)),
                           ("Q-learning", self.q)):
            scores = np.array([r.score for r in runs])
            out.append({
                "route": name, "runs": len(runs), "score_mean": float(scores.mean()),
                "score_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "regret_mean": float(self.optimum - scores.mean()),
                "env_steps": runs[0].env_steps, "backups": runs[0].backups,
                "seconds_per_run": float(np.mean([r.seconds for r in runs])),
            })
        return out

    def table(self) -> str:
        """The comparison as text: one row per route, spread across seeds for the random ones."""
        header = (f"{'route':<18}{'runs':>5}  {'score mean':>12}{'sd':>8}{'min':>12}{'max':>12}"
                  f"  {'regret mean':>12}  {'env steps':>10}{'backups':>9}{'sec/run':>8}")
        lines = [f"optimum (value iteration) = {self.optimum:,.1f} tenge", header]
        for r in self.rows():
            lines.append(
                f"{r['route']:<18}{r['runs']:>5}  {r['score_mean']:>12,.1f}{r['score_sd']:>8,.1f}"
                f"{r['score_min']:>12,.1f}{r['score_max']:>12,.1f}  {r['regret_mean']:>12,.1f}"
                f"  {r['env_steps']:>10,}{r['backups']:>9,}{r['seconds_per_run']:>8.2f}")
        return "\n".join(lines)


def compare_strategies(params: ShopParams, gamma: float, *, backup: BackupFn,
                       choose_action: ChooseFn, update: UpdateFn, q_steps: int,
                       seeds: tuple[int, ...], sweep_episodes: int = 30,
                       alpha: float = 0.5, alpha_half_life: float = 50.0,
                       epsilon: float = 0.2) -> Comparison:
    """Run all three routes on the same shop, same gamma, same scoring rule.

    The two stochastic routes are repeated for every seed in ``seeds``, so the table shows how
    much a single run can be trusted. Nothing but the route differs between them.
    """
    if len(seeds) < 2:
        raise ValueError("give at least two seeds, otherwise there is no spread to report")
    vi = run_value_iteration(params, gamma, backup)
    sweep = tuple(run_order_up_to_sweep(params, gamma, sweep_episodes, s) for s in seeds)
    q = tuple(run_q_learning(params, gamma, q_steps, s, choose_action, update, alpha,
                             alpha_half_life, epsilon) for s in seeds)
    return Comparison(params, gamma, vi, sweep, q)


# ---------------------------------------------------------------------------------------------
# Self-checks for the cells you write. They test your function on small cases whose answer is
# known from the rules of the shop, and raise AssertionError with a hint when it is wrong.
# ---------------------------------------------------------------------------------------------


def check_bellman_backup(backup: BackupFn, params: ShopParams | None = None,
                         gamma: float = 0.95) -> None:
    """Test a value-iteration backup against the rules of the shop.

    Three tests: with ``v = 0`` the lookahead is the expected reward; with a constant ``v = c``
    it is that reward plus ``gamma * c`` (the rows of ``P`` sum to one); and for a lumpy ``v`` it
    equals the expectation over the demand distribution, computed from ``day_outcome`` directly.

    Raises:
        AssertionError: With a hint, when a test fails.
    """
    params = params or ShopParams()
    p_next, reward = transition_model(params)
    mask = legal_mask(params)
    n = params.n_states

    q0 = np.asarray(backup(p_next, reward, mask, np.zeros(n), gamma), dtype=float)
    assert q0.shape == mask.shape, f"q must have shape {mask.shape}, got {q0.shape}"
    assert np.allclose(q0[mask], reward[mask]), (
        "with v = 0 the one-step lookahead should equal the expected reward R[s, a]")

    c = 1000.0
    q1 = np.asarray(backup(p_next, reward, mask, np.full(n, c), gamma), dtype=float)
    assert np.allclose(q1[mask], (reward + gamma * c)[mask]), (
        "with v = 1000 everywhere every legal q should be R[s, a] + gamma * 1000: "
        "check the discount and that you sum over the NEXT state")

    v = np.arange(n, dtype=float) ** 2 * 37.0
    q2 = np.asarray(backup(p_next, reward, mask, v, gamma), dtype=float)
    for stock, order in [(0, 3), (2, 0), (4, 4), (7, 1), (10, 0)]:
        expected = 0.0
        for demand, prob in enumerate(params.demand_probs):
            day = day_outcome(params, stock, order, demand)
            expected += prob * (day.reward + gamma * v[day.end_stock])
        assert abs(q2[stock, order] - expected) < 1e-6, (
            f"q[{stock}, {order}] = {q2[stock, order]:.3f}, but averaging the days over the "
            f"demand distribution gives {expected:.3f}: check which axis of P you sum over")
    print("check_bellman_backup: all three tests passed")


def check_q_learning(choose_action: ChooseFn, update: UpdateFn) -> None:
    """Test an epsilon-greedy ``choose_action`` and a Q-learning ``update`` on small cases.

    Raises:
        AssertionError: With a hint, when a test fails.
    """
    params = ShopParams()
    mask = legal_mask(params)
    rng = np.random.default_rng(0)
    q = np.zeros(mask.shape)
    q[~mask] = -np.inf

    q[3, :8] = [1.0, 2.0, 9.0, 4.0, 0.0, 3.0, 1.0, 2.0]
    greedy = int(choose_action(q, 3, 0.0, mask, rng))
    assert greedy == 2, f"with epsilon = 0 the best legal order of stock 3 is 2, got {greedy}"
    draws = [int(choose_action(q, 3, 1.0, mask, rng)) for _ in range(400)]
    assert all(mask[3, a] for a in draws), "with epsilon = 1 an order that does not fit was drawn"
    assert len(set(draws)) >= 5, "with epsilon = 1 the draws should spread over the legal orders"
    mixed = [int(choose_action(q, 3, 0.2, mask, rng)) for _ in range(2000)]
    share_greedy = mixed.count(2) / len(mixed)
    assert 0.8 <= share_greedy <= 0.93, (
        f"with epsilon = 0.2 the greedy order should be chosen about 80-90 % of the time "
        f"(0.8 plus its share of the random draws), got {share_greedy:.2f}")

    q = np.zeros(mask.shape)
    q[~mask] = -np.inf
    q[2, 1] = 10.0
    q[4, :7] = [0.0, 5.0, 20.0, 100.0, 40.0, 1.0, 2.0]
    before = q.copy()
    update(q, 2, 1, 7.0, 4, 0.1, 0.9, mask)
    assert abs(q[2, 1] - 18.7) < 1e-9, (
        f"Q(2, 1) was 10, reward 7, best Q of the next state 100, alpha 0.1, gamma 0.9: "
        f"the target is 7 + 0.9 * 100 = 97 and Q should move a tenth of the way, to 18.7; "
        f"got {q[2, 1]}")
    other = np.ones(q.shape, dtype=bool)
    other[2, 1] = False
    assert np.array_equal(q[other], before[other]), "update must change Q(state, action) only"
    print("check_q_learning: all tests passed")
