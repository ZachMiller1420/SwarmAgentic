from typing import Dict, List, Tuple
from .agent_spec import AgentSystem
import os
import re

# Cached domain keyword set (optional)
_DOMAIN_KEYWORDS: List[str] = []
_DOMAIN_KEYWORDS_LOADED: bool = False


def _load_domain_keywords():
    global _DOMAIN_KEYWORDS, _DOMAIN_KEYWORDS_LOADED
    if _DOMAIN_KEYWORDS_LOADED:
        return
    _DOMAIN_KEYWORDS_LOADED = True
    try:
        csv = os.environ.get("PSO_DOMAIN_KEYWORDS", "").strip()
        if csv:
            words = [w.strip().lower() for w in csv.split(",") if len(w.strip()) > 3]
            _DOMAIN_KEYWORDS = list(dict.fromkeys(words))
            return
        path = os.environ.get("PSO_DOMAIN_TEXT_PATH", "").strip()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                # Extract candidate keywords by frequency, filtering short/stop-like tokens
                tokens = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", txt)]
                # Simple frequency count
                freq: Dict[str, int] = {}
                for t in tokens:
                    freq[t] = freq.get(t, 0) + 1
                # Keep top-N frequent domain terms
                top_n = int(os.environ.get("PSO_DOMAIN_TOP_N", "40"))
                _DOMAIN_KEYWORDS = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
            except Exception:
                _DOMAIN_KEYWORDS = []
        else:
            _DOMAIN_KEYWORDS = []
    except Exception:
        _DOMAIN_KEYWORDS = []


def evaluate_system(system: AgentSystem, tasks: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic fitness: higher is better.
    - Coverage: roles/workflow cover task keywords
    - Balance: avoid role redundancy; ensure verify step exists
    - Length penalties: too long workflows are penalized
    """
    text = (" ".join([r.name for r in system.roles]) + " " +
            " ".join([" ".join(r.responsibilities) for r in system.roles]) + " " +
            " ".join(system.workflow)).lower()

    # Keyword coverage across tasks
    keywords = set()
    for t in tasks:
        for w in t.lower().replace("/", " ").split():
            if len(w) > 3:
                keywords.add(w)

    hits = sum(1 for k in keywords if k in text)
    coverage = hits / max(1, len(keywords))

    # Role balance: prefer 3-6 roles, unique names (original behavior)
    unique_names = len(set([r.name.lower() for r in system.roles]))
    role_count = len(system.roles)
    redundancy_penalty = 0.0 if unique_names == role_count else 0.2
    size_bonus = 1.0 if 3 <= role_count <= 6 else 0.7

    # Verification: bonus if any verify/audit/check step
    verify_bonus = 0.15 if any("verify" in s.lower() or "check" in s.lower() or "audit" in s.lower() for s in system.workflow) else 0.0

    # Workflow length penalty
    wf_len = len(system.workflow)
    wf_penalty = 0.0
    if wf_len == 0:
        wf_penalty = 0.4
    elif wf_len > 10:
        wf_penalty = 0.2

    # Aggregate fitness
    # Optional domain relevance term (derived from training corpus if provided)
    _load_domain_keywords()
    domain_coverage = 0.0
    if _DOMAIN_KEYWORDS:
        dhits = sum(1 for k in set(_DOMAIN_KEYWORDS) if k in text)
        domain_coverage = dhits / max(1, len(set(_DOMAIN_KEYWORDS)))

    domain_weight = float(os.environ.get("PSO_DOMAIN_WEIGHT", "0.2")) if _DOMAIN_KEYWORDS else 0.0

    base = 0.5 * coverage + 0.2 * size_bonus + verify_bonus + domain_weight * domain_coverage
    fitness = max(0.0, min(1.0, base - redundancy_penalty - wf_penalty))

    metrics = {
        "coverage": coverage,
        "size_bonus": size_bonus,
        "verify_bonus": verify_bonus,
        "redundancy_penalty": redundancy_penalty,
        "wf_penalty": wf_penalty,
        "domain_coverage": domain_coverage,
    }
    return fitness, metrics
