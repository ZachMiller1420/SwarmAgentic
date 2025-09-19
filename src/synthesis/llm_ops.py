"""
Optional LLM-powered operators for text-based PSO over agent specs.
Uses OpenAI-compatible API if OPENAI_API_KEY is set. Falls back gracefully.
Includes lightweight logging for debugging call/response outcomes.
"""

import json
import os
from typing import Optional, Tuple
import requests
import logging
import time

from .agent_spec import AgentSystem, AgentRole

_logger = logging.getLogger(__name__)
_last_call_ts: float = 0.0
_cb_open_until: float = 0.0
_cb_errors: int = 0

def _circuit_is_open() -> bool:
    try:
        return time.time() < _cb_open_until
    except Exception:
        return False

def _open_circuit():
    global _cb_open_until, _cb_errors
    try:
        cooldown = float(os.environ.get("OPENAI_CB_COOLDOWN_SEC", "60"))
    except Exception:
        cooldown = 60.0
    _cb_open_until = time.time() + max(1.0, cooldown)
    _cb_errors = 0
    try:
        _logger.warning(f"LLM circuit opened for {cooldown:.1f}s due to repeated errors/rate limits")
    except Exception:
        pass

def _respect_rate_limit():
    global _last_call_ts
    try:
        min_interval = float(os.environ.get("OPENAI_MIN_INTERVAL_SEC", "0.5"))
    except Exception:
        min_interval = 0.5
    now = time.time()
    wait = _last_call_ts + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


def _extract_json_block(s: str) -> Optional[str]:
    """Best-effort extraction of the first top-level JSON object from text."""
    if not s:
        return None
    try:
        # Fast path: direct JSON
        json.loads(s)
        return s
    except Exception:
        pass
    # Scan for a balanced {...}
    start = s.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
        start = s.find('{', start + 1)
    return None


def _coerce_system_from_json(s: str) -> Optional[AgentSystem]:
    try:
        data_str = _extract_json_block(s) or s
        data = json.loads(data_str)
        roles = []
        for r in data.get("roles", []):
            if isinstance(r, dict) and "name" in r:
                roles.append(AgentRole(
                    name=r.get("name", "Agent"),
                    responsibilities=list(r.get("responsibilities", [])),
                    tools=list(r.get("tools", [])),
                ))
        workflow = [str(x) for x in data.get("workflow", [])]
        if roles or workflow:
            return AgentSystem(roles=roles, workflow=workflow)
    except Exception:
        return None
    return None


def _openai_chat(messages, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 600) -> Optional[str]:
    """Call OpenAI chat completion via HTTP. Returns content string or None."""
    # Circuit breaker: skip calls during cooldown
    if _circuit_is_open():
        try:
            remain = max(0.0, _cb_open_until - time.time())
            _logger.info(f"LLM circuit open; skipping call (remaining {remain:.1f}s)")
        except Exception:
            pass
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            _logger.info("LLM disabled: OPENAI_API_KEY not set")
        except Exception:
            pass
        return None
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.environ.get("OPENAI_MODEL", model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Encourage structured JSON responses when supported
    if os.environ.get("OPENAI_FORCE_JSON", "1") == "1":
        try:
            payload["response_format"] = {"type": "json_object"}
        except Exception:
            pass
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
    base_backoff = float(os.environ.get("OPENAI_BACKOFF_BASE", "0.5"))
    cb_threshold = int(os.environ.get("OPENAI_CB_THRESHOLD", "3"))
    open_on_429 = os.environ.get("OPENAI_CB_ON_429", "1") == "1"
    for attempt in range(max_retries + 1):
        try:
            _respect_rate_limit()
            try:
                _logger.debug(f"Calling OpenAI chat: model={payload.get('model')} messages={len(messages)} attempt={attempt}")
            except Exception:
                pass
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                delay = float(retry_after) if retry_after and retry_after.isdigit() else base_backoff * (2 ** attempt)
                _logger.warning(f"LLM 429 Too Many Requests; backing off for {delay:.2f}s")
                # Count errors and potentially open circuit immediately on 429s
                global _cb_errors
                _cb_errors += 1
                if open_on_429 and _cb_errors >= cb_threshold:
                    _open_circuit()
                    return None
                time.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            try:
                _logger.debug(f"LLM response received (len={len(content or '')})")
            except Exception:
                pass
            # Success: reset error counter
            _cb_errors = 0
            return content
        except Exception:
            try:
                _logger.warning("LLM API call failed; falling back", exc_info=True)
            except Exception:
                pass
            # Count errors; open circuit if threshold exceeded
            _cb_errors += 1
            if _cb_errors >= cb_threshold:
                _open_circuit()
                return None
            if attempt < max_retries:
                delay = base_backoff * (2 ** attempt)
                time.sleep(delay)
                continue
            return None


def llm_mutate_system(base: AgentSystem, objective: str, pbest_text: str, gbest_text: str) -> Optional[AgentSystem]:
    """
    Ask an LLM to refine the base agent-system toward the objective, taking into
    account personal best and global best text specs. Returns a parsed AgentSystem
    if successful, otherwise None.
    """
    sys_prompt = (
        "You are optimizing an agent-team design. Produce a JSON object with keys "
        "'roles' (array of {name, responsibilities, tools}) and 'workflow' (array of strings). "
        "Focus on coverage of the objective, include a verification step, and avoid redundant roles."
    )
    user_prompt = (
        f"Objective:\n{objective}\n\n"
        f"Current spec (BASE):\n{base.to_text()}\n\n"
        f"Personal best (PBEST):\n{pbest_text}\n\n"
        f"Global best (GBEST):\n{gbest_text}\n\n"
        "Return ONLY valid JSON. Do not include any prose, code fences, or comments."
    )
    _logger.info("Requesting LLM mutation for agent-system optimization")
    # Use a lower temperature for structured output unless overridden
    try:
        temp = float(os.environ.get("OPENAI_TEMPERATURE", "0.1"))
    except Exception:
        temp = 0.1
    content = _openai_chat([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=temp)
    if content is None:
        try:
            _logger.info("No LLM content returned; using rule-based mutation")
        except Exception:
            pass
        return None
    sys_obj = _coerce_system_from_json(content or "{}")
    if sys_obj is None:
        try:
            preview = (content or "").strip().replace("\n", " ")[:160]
            _logger.info(f"LLM content parse failed; preview='{preview}'")
        except Exception:
            pass
    return sys_obj


def llm_mutate_batch(base: AgentSystem, objective: str, pbest_text: str, gbest_text: str, count: int = 3) -> Optional[list]:
    """
    Ask the LLM for multiple improved candidates at once. Returns a list of AgentSystem or None.
    Uses a JSON wrapper object: {"candidates": [ {...}, {...} ]} for robust parsing.
    """
    sys_prompt = (
        "You are optimizing an agent-team design. Return a JSON object with key 'candidates' "
        "mapping to an array of 2-5 items. Each item is a JSON object with keys 'roles' (array of {name, responsibilities, tools}) "
        "and 'workflow' (array of strings). Focus on coverage of the objective, include a verification step, and avoid redundant roles."
    )
    user_prompt = (
        f"Objective:\n{objective}\n\n"
        f"Current best (GBEST):\n{gbest_text}\n\n"
        f"Personal best (PBEST exemplar):\n{pbest_text}\n\n"
        f"Produce {max(1, count)} diverse candidates. Return ONLY JSON with a 'candidates' array; no prose."
    )
    try:
        temp = float(os.environ.get("OPENAI_TEMPERATURE", "0.1"))
    except Exception:
        temp = 0.1
    content = _openai_chat([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=temp, max_tokens=1000)
    if not content:
        try:
            _logger.info("No LLM content returned for batch; using rule-based mutation")
        except Exception:
            pass
        return None
    # Try to coerce
    try:
        wrapper_str = _extract_json_block(content) or content
        data = json.loads(wrapper_str)
        items = data.get("candidates", []) if isinstance(data, dict) else []
        result = []
        for item in items:
            try:
                roles = []
                for r in item.get("roles", []) if isinstance(item, dict) else []:
                    if isinstance(r, dict) and "name" in r:
                        roles.append(AgentRole(
                            name=r.get("name", "Agent"),
                            responsibilities=list(r.get("responsibilities", [])),
                            tools=list(r.get("tools", [])),
                        ))
                workflow = [str(x) for x in item.get("workflow", [])] if isinstance(item, dict) else []
                if roles or workflow:
                    result.append(AgentSystem(roles=roles, workflow=workflow))
            except Exception:
                continue
        return result or None
    except Exception:
        try:
            preview = (content or "").strip().replace("\n", " ")[:160]
            _logger.info(f"LLM batch parse failed; preview='{preview}'")
        except Exception:
            pass
        return None
