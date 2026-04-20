#!/usr/bin/env python3
"""
Generate short retrieval-oriented hints for dishes using an LLM (OpenRouter).

The hints are meant to be appended into ``dish_to_rich_text(..., retrieval_hint=...)``
for item-to-item retrieval experiments.

Hard constraints for the model:
- Use ONLY facts present in the provided JSON fields.
- Do NOT invent allergens, cuisines, ingredients, brands, or health claims.
- Output MUST be JSON: {"id": <int>, "retrieval_hint": "<= 240 chars English>"}

Auth/config (in priority order):
1) CLI flags: ``--api-key``, ``--api-base``
2) Environment variables: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``
3) Dotenv files (repo root overrides legacy backend keys if duplicated):
   - ``backend/.env`` (legacy): ``ORACLE_API_KEY``, ``ORACLE_API_BASE`` (same as ``data/generate_dishes_llm.py``)
   - ``.env`` in repo root (optional): ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` or the ``ORACLE_*`` keys

Example:
  python scripts/augment_dishes_retrieval_llm.py --dishes-parquet data/synthetic/dishes.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

BATCH_SIZE = 8
MODEL = "openai/gpt-4o-mini"

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH_BACKEND = REPO_ROOT / "backend" / ".env"
ENV_PATH_ROOT = REPO_ROOT / ".env"
DEFAULT_OUT = REPO_ROOT / "data" / "dish_retrieval_aug.json"
DEFAULT_PROGRESS = REPO_ROOT / "data" / "dish_retrieval_aug_progress.json"


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
    """Load ``backend/.env`` first, then overlay ``.env`` at repo root."""
    merged = _load_env(ENV_PATH_BACKEND)
    merged.update(_load_env(ENV_PATH_ROOT))
    return merged


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _facts_row(row: dict[str, Any]) -> dict[str, Any]:
    """Minimal facts payload for the LLM (keep small)."""
    ingredients = row.get("ingredients")
    if hasattr(ingredients, "tolist"):
        ingredients = ingredients.tolist()
    if ingredients is None:
        ingredients = []

    tags = row.get("tag_list")
    if hasattr(tags, "tolist"):
        tags = tags.tolist()
    if tags is None:
        tags = []

    return {
        "id": int(row["id"]),
        "name": str(row.get("name") or ""),
        "meal_time": str(row.get("meal_time") or ""),
        "cuisine_cluster": int(row.get("cuisine_cluster")) if row.get("cuisine_cluster") is not None else None,
        "price_tier": str(row.get("price_tier") or ""),
        "calories": float(row.get("calories") or 0.0),
        "protein_g": float(row.get("protein_g") or 0.0),
        "fat_g": float(row.get("fat_g") or 0.0),
        "carbs_g": float(row.get("carbs_g") or 0.0),
        "fiber_g": float(row.get("fiber_g") or 0.0),
        "top_ingredients": [str(x) for x in list(ingredients)[:12]],
        "tags": [str(t) for t in list(tags)[:12]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dishes-parquet", default=str(REPO_ROOT / "data" / "synthetic" / "dishes.parquet"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI-compatible API key. For many local servers you can pass 'ollama' or 'lm-studio'.",
    )
    parser.add_argument(
        "--api-base",
        default="",
        help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1 (Ollama) or http://localhost:1234/v1 (LM Studio).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument(
        "--max-dishes",
        type=int,
        default=0,
        help="If >0, only process the first N dishes (debug/smoke test).",
    )
    args = parser.parse_args()

    env = _load_repo_dotenv()
    api_key = (
        str(args.api_key).strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or env.get("OPENAI_API_KEY", "").strip()
        or env.get("ORACLE_API_KEY", "").strip()
    )
    api_base = (
        str(args.api_base).strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or env.get("OPENAI_BASE_URL", "").strip()
        or env.get("ORACLE_API_BASE", "").strip()
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        print(
            "ERROR: No API key found. Provide one via:\n"
            "  --api-key ...\n"
            "or set OPENAI_API_KEY\n"
            f"or put ORACLE_API_KEY / OPENAI_API_KEY into {ENV_PATH_ROOT} or {ENV_PATH_BACKEND}\n"
            "Many local OpenAI-compatible servers still require a non-empty key string."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    dishes_path = Path(args.dishes_parquet)
    if not dishes_path.is_file():
        raise FileNotFoundError(str(dishes_path))

    out_path = Path(args.out)
    progress_path = Path(args.progress)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dishes_path)
    if int(args.max_dishes) > 0:
        df = df.iloc[: int(args.max_dishes)].copy()
    rows = df.to_dict(orient="records")

    done: dict[str, Any] = {}
    if progress_path.exists():
        done = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(done, dict):
            done = {}

    system = (
        "You write compact retrieval hints for a food recommender embedding model.\n"
        "Return ONLY valid JSON with keys: id (int), retrieval_hint (string).\n"
        "Rules:\n"
        "- Use ONLY facts from the user message JSON.\n"
        "- No invented allergens/cuisines/ingredients.\n"
        "- retrieval_hint must be English, <= 240 characters, no newlines.\n"
        "- Prefer including: meal_time, cuisine_cluster id, 3-6 anchor ingredients from top_ingredients,\n"
        "  and 1-2 macro cues derived ONLY from the numeric fields.\n"
    )

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_ids = [int(r["id"]) for r in batch]
        if all(str(i) in done for i in batch_ids):
            continue

        for r in batch:
            rid = int(r["id"])
            if str(rid) in done:
                continue

            user = json.dumps(_facts_row(r), ensure_ascii=False)

            last_err: Exception | None = None
            for attempt in range(6):
                try:
                    resp = client.chat.completions.create(
                        model=str(args.model),
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    content = resp.choices[0].message.content or ""
                    obj = _extract_json_object(content)
                    hint = str(obj.get("retrieval_hint", "")).strip()
                    out_id = obj.get("id")
                    if out_id is None:
                        raise ValueError("missing id")
                    out_id = int(out_id)
                    if out_id != rid:
                        raise ValueError(f"id mismatch: expected {rid}, got {out_id}")
                    if not hint or len(hint) > 240:
                        raise ValueError(f"bad hint len={len(hint)}")
                    done[str(rid)] = hint
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.5 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed dish_id={rid}: {last_err}")

            progress_path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(float(args.sleep_s))

    # Final export as sorted list for stable diffs
    export = [{"id": int(k), "retrieval_hint": str(v)} for k, v in done.items()]
    export.sort(key=lambda x: x["id"])
    out_path.write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(export):,} hints -> {out_path}")


if __name__ == "__main__":
    main()
