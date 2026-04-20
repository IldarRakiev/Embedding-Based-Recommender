# src/llm_profile_reranker.py
"""LLM reranking for retrieval candidates using a short user profile.

This is meant for offline experiments (Kaggle/Colab/local) using an OpenAI-compatible
HTTP API (OpenRouter, local vLLM, Ollama with OpenAI shim, etc.).

The reranker expects **candidate IDs are closed-world**: the model must only reorder
IDs from the provided candidate list.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH_BACKEND = REPO_ROOT / "backend" / ".env"
ENV_PATH_ROOT = REPO_ROOT / ".env"


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_repo_dotenv() -> dict[str, str]:
    merged = _load_env(ENV_PATH_BACKEND)
    merged.update(_load_env(ENV_PATH_ROOT))
    return merged


def _as_plain_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def user_row_to_profile_text(row: dict[str, Any]) -> str:
    """Compact English profile string from a ``users.parquet`` row dict."""
    allergens = [str(a) for a in _as_plain_list(row.get("allergens")) if str(a).strip()]

    parts: list[str] = []
    parts.append(f"user_id={int(row['user_id'])}")
    if row.get("user_type"):
        parts.append(f"user_type={row.get('user_type')}")
    if row.get("archetype_name"):
        parts.append(f"archetype={row.get('archetype_name')}")
    if row.get("goal_type"):
        parts.append(f"goal={row.get('goal_type')}")
    if row.get("activity_level"):
        parts.append(f"activity={row.get('activity_level')}")
    if row.get("gender"):
        parts.append(f"gender={row.get('gender')}")
    if row.get("age") is not None:
        parts.append(f"age={row.get('age')}")
    if row.get("height_cm") is not None:
        parts.append(f"height_cm={row.get('height_cm')}")
    if row.get("weight_kg") is not None:
        parts.append(f"weight_kg={row.get('weight_kg')}")
    if row.get("has_diabetes") is not None:
        parts.append(f"has_diabetes={bool(row.get('has_diabetes'))}")
    if row.get("price_level"):
        parts.append(f"price_level={row.get('price_level')}")

    if row.get("kcal_target") is not None:
        parts.append(f"kcal_target={row.get('kcal_target')}")
    if row.get("protein_target_g") is not None:
        parts.append(f"protein_target_g={row.get('protein_target_g')}")
    if row.get("fat_target_g") is not None:
        parts.append(f"fat_target_g={row.get('fat_target_g')}")
    if row.get("carbs_target_g") is not None:
        parts.append(f"carbs_target_g={row.get('carbs_target_g')}")

    if allergens:
        parts.append("allergens=" + ", ".join(allergens))

    return " | ".join(parts)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", text)
    if not m:
        raise ValueError(f"No JSON found in model output: {text[:200]!r}")
    return json.loads(m.group(1))


def _parse_ranked_ids(obj: Any) -> list[int]:
    if isinstance(obj, dict):
        for k in ("ranked_ids", "ranked_dish_ids", "ids", "order"):
            if k in obj and isinstance(obj[k], list):
                return [int(x) for x in obj[k]]
        # Sometimes models nest under "result"
        if isinstance(obj.get("result"), dict):
            return _parse_ranked_ids(obj["result"])
        raise ValueError(f"Unrecognized JSON object keys: {list(obj.keys())[:20]}")
    if isinstance(obj, list):
        return [int(x) for x in obj]
    raise ValueError(f"Unexpected JSON type: {type(obj)}")


class LLMProfileReranker:
    """Rerank dense retrieval candidates with an LLM using (profile, query dish, candidates)."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        env = _load_repo_dotenv()
        key = (
            (api_key or "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or env.get("OPENAI_API_KEY", "").strip()
            or env.get("ORACLE_API_KEY", "").strip()
        )
        base = (
            (api_base or "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or env.get("OPENAI_BASE_URL", "").strip()
            or env.get("ORACLE_API_BASE", "").strip()
            or "https://openrouter.ai/api/v1"
        )
        self._api_key = key
        self._api_base = base
        self.model = model
        self.temperature = temperature
        self._client: OpenAI | None = OpenAI(api_key=key, base_url=base) if key else None

    @property
    def available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def rerank(
        self,
        profile_text: str,
        query_dish_text: str,
        candidates: list[tuple[int, str]],
        top_k: int = 10,
    ) -> list[int]:
        """Return up to ``top_k`` dish IDs in LLM order (closed-world over candidates)."""
        if not candidates:
            return []
        if not self.available or self._client is None:
            return [c[0] for c in candidates[:top_k]]

        cand_lines: list[str] = []
        allowed: set[int] = set()
        for did, txt in candidates:
            allowed.add(int(did))
            cand_lines.append(f"- id={int(did)} :: {str(txt).strip()}")

        system = (
            "You rerank food dish candidates for the SAME user.\n"
            "You will be given: USER_PROFILE, QUERY_DISH the user already likes, and CANDIDATES.\n"
            "Task: pick the candidates most likely to be co-preferred with QUERY_DISH for this user,\n"
            "using USER_PROFILE constraints (allergens, goals, macros, price) when applicable.\n"
            "Return ONLY valid JSON: {\"ranked_ids\": [<int>, ...]}.\n"
            "Rules:\n"
            "- ranked_ids MUST be a permutation of ALL candidate ids shown (same set, same length).\n"
            "- Do NOT invent ids.\n"
            "- Most likely items first.\n"
        )

        user = (
            "USER_PROFILE:\n"
            f"{profile_text}\n\n"
            "QUERY_DISH:\n"
            f"{query_dish_text}\n\n"
            "CANDIDATES (one per line):\n"
            + "\n".join(cand_lines)
        )

        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=float(self.temperature),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        ranked = _parse_ranked_ids(_extract_json(content))

        # Sanitize: keep only allowed, unique, then append missing in original order
        out: list[int] = []
        seen: set[int] = set()
        for x in ranked:
            xi = int(x)
            if xi in allowed and xi not in seen:
                out.append(xi)
                seen.add(xi)
        for did, _ in candidates:
            di = int(did)
            if di in allowed and di not in seen:
                out.append(di)
                seen.add(di)

        return out[: int(top_k)]
