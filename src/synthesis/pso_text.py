from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable, Optional
import random
import logging
import os

from .agent_spec import AgentSystem, AgentRole
from .eval import evaluate_system
from .llm_ops import llm_mutate_system, llm_mutate_batch


@dataclass
class PSOResult:
    best_system: AgentSystem
    best_fitness: float
    history: List[Dict[str, Any]]


class PSOSwarmSynthesizer:
    """
    Minimal, rule-based PSO over textual agent specs.
    No external LLMs; we simulate 'velocity' with deterministic mutations.
    """

    def __init__(self, population_size: int = 6, iterations: int = 8, seed: int = 42, use_llm: bool = False):
        self.population_size = population_size
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.use_llm = use_llm
        # LLM debugging counters
        self.llm_calls: int = 0
        self.llm_accepts: int = 0
        self.llm_noops: int = 0  # calls that did not yield an accepted mutation
        self._accepted_this_iter: int = 0
        self._logger = logging.getLogger(__name__)
        # LLM control flags
        try:
            self.llm_gbest_only = os.environ.get('LLM_PSO_GBEST_ONLY', '1') == '1'
            self.llm_prob = max(0.0, min(1.0, float(os.environ.get('LLM_PSO_PROB', '0.3'))))
            self.llm_max_calls_per_iter = max(0, int(os.environ.get('LLM_PSO_MAX_CALLS_PER_ITER', '3')))
        except Exception:
            self.llm_gbest_only = True
            self.llm_prob = 0.3
            self.llm_max_calls_per_iter = 3
        self._calls_this_iter: int = 0
        # Simple in-memory cache to avoid duplicate LLM calls for same prompt context
        self._llm_cache: Dict[str, AgentSystem] = {}

        # Expanded responsibilities to better match Software Delivery and Research task keywords
        self.role_pool = [
            (
                "Coordinator",
                [
                    "orchestrate",
                    "assign tasks",
                    "monitor",
                    "coordinate sprints",
                    "manage release",
                    "schedule code review",
                ],
                ["dashboard"],
            ),
            (
                "Planner",
                [
                    "plan",
                    "decompose",
                    "prioritize",
                    "collect requirements",
                    "sprint planning",
                    "design experiments",
                ],
                ["planner"],
            ),
            (
                "Executor",
                [
                    "execute",
                    "call tools",
                    "collect results",
                    "implement features",
                    "write tests",
                    "fix bugs",
                    "run experiments",
                    "preprocess data",
                    "train models",
                ],
                ["tools"],
            ),
            (
                "Verifier",
                [
                    "verify",
                    "cross-check",
                    "evaluate",
                    "verify release",
                    "run regression tests",
                    "evaluate metrics",
                    "peer review checks",
                ],
                ["checker"],
            ),
            (
                "Researcher",
                [
                    "search",
                    "gather info",
                    "summarize",
                    "literature review",
                    "define hypotheses",
                    "design experiments",
                    "collect datasets",
                    "analyze results",
                ],
                ["search"],
            ),
            (
                "Critic",
                [
                    "critique",
                    "identify flaws",
                    "suggest fixes",
                    "code review",
                    "find bugs",
                    "statistical critique",
                ],
                ["lint"],
            ),
        ]

        # Include delivery and research phrasing so coverage improves for those task sets
        self.wf_pool = [
            # General
            "collect requirements",
            "plan tasks",
            "execute plan",
            "verify results",
            "refine and iterate",
            # Software Delivery
            "plan sprints",
            "implement features",
            "write tests",
            "code review",
            "verify release",
            "triage bugs",
            "reproduce issues",
            "implement fixes",
            "add regression tests",
            "verify patch",
            # Research
            "literature review",
            "summarize findings",
            "define hypotheses",
            "design experiments",
            "analyze results",
            "verify conclusions",
            "collect datasets",
            "clean and preprocess",
            "train models",
            "evaluate metrics",
            "document results",
            "peer review",
        ]

    def _random_role(self) -> AgentRole:
        name, resp, tools = self.rng.choice(self.role_pool)
        # Subsample responsibilities/tools to vary
        r = self.rng.sample(resp, k=self.rng.randint(1, len(resp)))
        t = self.rng.sample(tools, k=self.rng.randint(1, len(tools)))
        return AgentRole(name=name, responsibilities=r, tools=t)

    def _system_signature(self, system: AgentSystem) -> str:
        """A normalized signature used to detect duplicates.
        Uses role names (sorted) + workflow (in order) for a coarse uniqueness test.
        """
        try:
            role_names = sorted([r.name.lower().strip() for r in system.roles])
            wf = [w.lower().strip() for w in system.workflow]
            return "|".join(role_names) + "::" + "->".join(wf)
        except Exception:
            return system.to_text()

    def _sanitize_unique_roles(self, system: AgentSystem) -> AgentSystem:
        """Drop duplicate role names, keeping the first occurrence (stability)."""
        seen = set()
        roles: List[AgentRole] = []
        for r in system.roles:
            if r.name.lower() in seen:
                continue
            seen.add(r.name.lower())
            roles.append(r)
        if len(roles) == len(system.roles):
            return system
        return AgentSystem(roles=roles, workflow=list(system.workflow))

    def _random_system(self) -> AgentSystem:
        # Original behavior: moderate team size and avoid duplicate role names
        n_roles = self.rng.randint(2, 5)
        roles = []
        names_seen = set()
        for _ in range(n_roles):
            role = self._random_role()
            # avoid immediate duplicate names
            if role.name in names_seen:
                continue
            names_seen.add(role.name)
            roles.append(role)
        # Ensure we never sample more workflow steps than available in the pool
        max_wf = max(1, len(self.wf_pool))
        wf_len = self.rng.randint(3, min(6, max_wf))
        # Randomly pick distinct workflow steps up to wf_len (capped by pool size)
        workflow = self.rng.sample(self.wf_pool, k=min(wf_len, len(self.wf_pool)))
        return AgentSystem(roles=roles, workflow=workflow)

    def _mutate(self, system: AgentSystem, guidance: Dict[str, float], pbest_text: str, gbest_text: str, objective: str) -> AgentSystem:
        new = AgentSystem(roles=[AgentRole(r.name, list(r.responsibilities), list(r.tools)) for r in system.roles],
                          workflow=list(system.workflow))

        ops = []
        # If coverage is low, try adding a role or workflow step
        if guidance.get("coverage", 0.0) < 0.6:
            ops.append("add_role")
            ops.append("add_workflow")
        # If no verify bonus, try adding verifier or verify step
        if guidance.get("verify_bonus", 0.0) < 0.1:
            ops.append("ensure_verify")
        # If redundancy penalty exists, remove duplicate role
        if guidance.get("redundancy_penalty", 0.0) > 0.0:
            ops.append("dedupe_roles")
        # Always consider minor shuffle
        ops.append("shuffle_workflow")
        # Diversity ops: replace or alter roles/tools/workflow occasionally
        ops.extend(["replace_role", "alter_role_resp", "alter_role_tools", "reshape_workflow"])

        # LLM-driven mutation on a per-candidate basis (disabled when gbest-only is active)
        if self.use_llm and not getattr(self, 'llm_gbest_only', True) and getattr(self, '_calls_this_iter', 0) < getattr(self, 'llm_max_calls_per_iter', 3):
            # lazily ensure controls exist (in case of older instances)
            if not hasattr(self, 'llm_prob'):
                try:
                    self.llm_prob = max(0.0, min(1.0, float(os.environ.get('LLM_PSO_PROB', '0.3'))))
                    self.llm_max_calls_per_iter = max(0, int(os.environ.get('LLM_PSO_MAX_CALLS_PER_ITER', '3')))
                except Exception:
                    self.llm_prob = 0.3
                    self.llm_max_calls_per_iter = 3
            if self.rng.random() < self.llm_prob:
                try:
                    self._calls_this_iter = getattr(self, '_calls_this_iter', 0) + 1
                    self.llm_calls += 1
                    try:
                        self._logger.debug("Invoking LLM mutation operator for text PSO")
                    except Exception:
                        pass
                    mutated = llm_mutate_system(new, objective, pbest_text, gbest_text)
                    if mutated:
                        self.llm_accepts += 1
                        self._accepted_this_iter += 1
                        try:
                            self._logger.info("LLM mutation accepted: updating candidate via LLM proposal")
                        except Exception:
                            pass
                        return mutated
                except Exception:
                    # If anything goes wrong, silently fall back to rule-based ops
                    try:
                        self._logger.warning("LLM mutation error; falling back to rule-based ops", exc_info=True)
                    except Exception:
                        pass
                # No usable LLM mutation this time
                self.llm_noops += 1

        # Sample 1-2 rule-based ops (with diversity operators included)
        for op in self.rng.sample(ops, k=min(len(ops), self.rng.randint(1, 2))):
            if op == "add_role":
                new.roles.append(self._random_role())
            elif op == "add_workflow":
                candidate = self.rng.choice(self.wf_pool)
                if candidate not in new.workflow:
                    new.workflow.append(candidate)
            elif op == "ensure_verify":
                # add verifier role if missing
                if not any(r.name.lower() == "verifier" for r in new.roles):
                    new.roles.append(AgentRole("Verifier", ["verify", "cross-check"], ["checker"]))
                # add verify step if missing
                if not any("verify" in s.lower() for s in new.workflow):
                    new.workflow.append("verify results")
            elif op == "dedupe_roles":
                seen = set()
                deduped = []
                for r in new.roles:
                    if r.name.lower() not in seen:
                        seen.add(r.name.lower())
                        deduped.append(r)
                new.roles = deduped
            elif op == "shuffle_workflow" and len(new.workflow) > 1:
                i = self.rng.randrange(len(new.workflow))
                j = self.rng.randrange(len(new.workflow))
                new.workflow[i], new.workflow[j] = new.workflow[j], new.workflow[i]
            elif op == "replace_role" and new.roles:
                idx = self.rng.randrange(len(new.roles))
                # Replace with a new role ensuring name uniqueness when possible
                seen = {r.name for r in new.roles}
                for _ in range(5):
                    cand = self._random_role()
                    if cand.name not in seen or cand.name == new.roles[idx].name:
                        new.roles[idx] = cand
                        break
            elif op == "alter_role_resp" and new.roles:
                idx = self.rng.randrange(len(new.roles))
                # Reshuffle responsibilities length
                # Find role template by name to get pool
                pool = next((resp for (name, resp, _tools) in self.role_pool if name == new.roles[idx].name), None)
                if pool:
                    new.roles[idx].responsibilities = self.rng.sample(pool, k=self.rng.randint(1, len(pool)))
            elif op == "alter_role_tools" and new.roles:
                idx = self.rng.randrange(len(new.roles))
                pool = next((tools for (name, _resp, tools) in self.role_pool if name == new.roles[idx].name), None)
                if pool:
                    new.roles[idx].tools = self.rng.sample(pool, k=self.rng.randint(1, len(pool)))
            elif op == "reshape_workflow":
                # Rebuild workflow from pool (distinct steps)
                max_wf = max(1, len(self.wf_pool))
                wf_len = self.rng.randint(3, min(6, max_wf))
                new.workflow = self.rng.sample(self.wf_pool, k=min(wf_len, len(self.wf_pool)))

        return new

    def run(self, tasks: List[str], on_iteration: Optional[Callable[[int, List[Tuple[AgentSystem, float, Dict[str, float]]], Tuple[AgentSystem, float]], None]] = None) -> PSOResult:
        # Initialize population with uniqueness by signature
        population: List[AgentSystem] = []
        seen_sigs = set()
        while len(population) < self.population_size:
            cand = self._random_system()
            sig = self._system_signature(cand)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            population.append(cand)
        pbest = []  # list of (fitness, system, metrics)
        gbest_fit = -1.0
        gbest_sys = None
        history: List[Dict[str, Any]] = []

        # Evaluate initial population
        for sys in population:
            fit, metrics = evaluate_system(sys, tasks)
            pbest.append((fit, sys, metrics))
            if fit > gbest_fit:
                gbest_fit, gbest_sys = fit, sys

        # Iterate
        for it in range(self.iterations):
            # reset per-iteration LLM acceptance/call counters
            self._accepted_this_iter = 0
            self._calls_this_iter = 0

            # Optional: late-stage + gbest-only LLM refinement once per iteration
            try:
                late_n = max(0, int(os.environ.get('LLM_PSO_LATE_STAGE_N', '3')))
            except Exception:
                late_n = 3
            late_stage = (it >= max(0, self.iterations - late_n))
            if self.use_llm and late_stage and getattr(self, 'llm_gbest_only', True) and gbest_sys is not None and self._calls_this_iter < self.llm_max_calls_per_iter:
                try:
                    base_text = gbest_sys.to_text()
                    objective = ", ".join(tasks)
                    cache_key = f"gbest::{hash(objective)}::{hash(base_text)}"
                    mutated = self._llm_cache.get(cache_key)
                    called = False
                    if mutated is None and self.rng.random() < self.llm_prob:
                        self.llm_calls += 1
                        self._calls_this_iter += 1
                        called = True
                    mutated = llm_mutate_system(gbest_sys, objective, base_text, base_text)
                    if mutated is not None:
                        mutated = self._sanitize_unique_roles(mutated)
                        if mutated is not None:
                            self._llm_cache[cache_key] = mutated
                    if mutated is not None:
                        cfit, cmetrics = evaluate_system(mutated, tasks)
                        # Inject into population by replacing worst pbest
                        worst_idx = min(range(len(pbest)), key=lambda i: pbest[i][0]) if pbest else None
                        if worst_idx is not None and cfit >= pbest[worst_idx][0]:
                            pbest[worst_idx] = (cfit, mutated, cmetrics)
                        if cfit > gbest_fit:
                            gbest_fit, gbest_sys = cfit, mutated
                        self.llm_accepts += 1
                        self._accepted_this_iter += 1
                    else:
                        if called:
                            self.llm_noops += 1
                except Exception:
                    # Any error in gbest LLM path -> count and continue
                    self.llm_noops += 1

            # Optional: late-stage batch proposals to amortize one call across multiple injections
            try:
                batch_on = os.environ.get('LLM_PSO_BATCH', '1') == '1'
                batch_count = max(1, int(os.environ.get('LLM_PSO_BATCH_COUNT', '3')))
            except Exception:
                batch_on = True
                batch_count = 3
            if self.use_llm and late_stage and self._calls_this_iter < self.llm_max_calls_per_iter and batch_on and gbest_sys is not None:
                try:
                    base_text = gbest_sys.to_text()
                    objective = ", ".join(tasks)
                    # limit to one batch call per iter
                    if self._calls_this_iter < self.llm_max_calls_per_iter and self.rng.random() < self.llm_prob:
                        self.llm_calls += 1
                        self._calls_this_iter += 1
                        batch = llm_mutate_batch(gbest_sys, objective, base_text, base_text, count=batch_count)
                        if batch:
                            # Evaluate and inject best few
                            scored = []
                            for cand in batch:
                                try:
                                    cand = self._sanitize_unique_roles(cand)
                                    f, m = evaluate_system(cand, tasks)
                                    scored.append((f, cand, m))
                                except Exception:
                                    continue
                            scored.sort(key=lambda x: x[0], reverse=True)
                            injected = 0
                            for f, cand, m in scored:
                                worst_idx = min(range(len(pbest)), key=lambda i: pbest[i][0]) if pbest else None
                                if worst_idx is None:
                                    break
                                if f >= pbest[worst_idx][0]:
                                    pbest[worst_idx] = (f, cand, m)
                                    injected += 1
                                    if f > gbest_fit:
                                        gbest_fit, gbest_sys = f, cand
                            if injected > 0:
                                self.llm_accepts += injected
                                self._accepted_this_iter += injected
                            else:
                                self.llm_noops += 1
                        else:
                            self.llm_noops += 1
                except Exception:
                    self.llm_noops += 1
            new_pop = []
            for idx, (fit, sys, metrics) in enumerate(pbest):
                # Influence from personal best and global best: choose one to copy/mutate towards
                base = sys if self.rng.random() < 0.5 else gbest_sys
                pbest_text = sys.to_text()
                gbest_text = (gbest_sys.to_text() if gbest_sys else "")
                objective = ", ".join(tasks)
                # Create child and enforce uniqueness against current pbest signatures
                attempt = 0
                while True:
                    child = self._mutate(base, metrics, pbest_text, gbest_text, objective)
                    child = self._sanitize_unique_roles(child)
                    sig = self._system_signature(child)
                    # If duplicate of any pbest system, try again up to 3 attempts
                    if sig not in {self._system_signature(sys) for (_f, sys, _m) in pbest} or attempt >= 3:
                        break
                    attempt += 1
                cfit, cmetrics = evaluate_system(child, tasks)
                # Update personal best
                if cfit >= fit:
                    pbest[idx] = (cfit, child, cmetrics)
                    fit, sys, metrics = cfit, child, cmetrics
                new_pop.append(sys)
                # Update global best
                if fit > gbest_fit:
                    gbest_fit, gbest_sys = fit, sys

            population = new_pop
            history.append({
                "iteration": it + 1,
                "gbest_fitness": gbest_fit,
                "llm_calls_total": self.llm_calls,
                "llm_accepts_total": self.llm_accepts,
                "llm_accepts_this_iter": self._accepted_this_iter,
                "llm_noops_total": self.llm_noops,
            })

            if on_iteration:
                pop_triplets = [(s, f, m) for (f, s, m) in [(pf[0], pf[1], pf[2]) for pf in pbest]]
                # Provide standard callback args; extra LLM stats can be read from this instance
                on_iteration(it + 1, pop_triplets, (gbest_sys, gbest_fit))

        return PSOResult(best_system=gbest_sys, best_fitness=gbest_fit, history=history)
