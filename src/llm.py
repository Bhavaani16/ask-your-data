"""LLM adapter backed by Hugging Face Inference Providers.

Turns a plain-English question into a dialect-correct SELECT using an open
model hosted on Hugging Face's serverless API (no local GPU needed).

Public API is unchanged from the original OpenAI version:
    generate_sql(question, schema, history, ...)
    build_prompt_preview(question, schema, history, ...)
    SQLResult

Model choice:
    Default is ``Qwen/Qwen2.5-Coder-32B-Instruct`` — strong at SQL and widely
    available on HF Inference. Override with the ``HF_MODEL`` env var.

Auth:
    Set ``HF_TOKEN`` in the environment (or `HUGGINGFACEHUB_API_TOKEN``,
    which ``huggingface_hub`` also picks up automatically). Get one free at
    https://huggingface.co/settings/tokens.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from huggingface_hub import InferenceClient

from .database import Schema


DEFAULT_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

# Optional: pin a specific provider (e.g. "hf-inference", "together",
# "fireworks-ai"). When unset, huggingface_hub picks one that serves the model.
DEFAULT_PROVIDER: Optional[str] = os.getenv("HF_PROVIDER") or None


_DIALECT_HINTS: dict[str, str] = {
    "sqlite":     "Use SQLite syntax. Dates are TEXT in ISO format; use date(), strftime().",
    "postgresql": "Use PostgreSQL syntax. Prefer DATE_TRUNC, TO_CHAR, INTERVAL, ILIKE for case-insensitive matches. Double-quote identifiers only if they need it.",
    "mysql":      "Use MySQL syntax. Prefer DATE_FORMAT, CURDATE(), INTERVAL. Quote identifiers with backticks only if they need it.",
    "mariadb":    "Use MariaDB/MySQL syntax. Prefer DATE_FORMAT, CURDATE(), INTERVAL.",
    "mssql":      "Use Microsoft SQL Server / T-SQL syntax. For row limits use TOP N (not LIMIT). Prefer DATEPART, DATEADD, GETDATE(), FORMAT().",
    "snowflake":  "Use Snowflake syntax. Prefer DATE_TRUNC, DATEADD, CURRENT_DATE(). Identifiers are case-insensitive unless quoted.",
    "bigquery":   "Use Google BigQuery Standard SQL. Prefer DATE_TRUNC, FORMAT_DATE, CURRENT_DATE(). Reference tables as `project.dataset.table` if provided; backtick-quote identifiers.",
    "oracle":     "Use Oracle SQL. Use FETCH FIRST N ROWS ONLY for row limits (not LIMIT). Prefer TO_CHAR, TO_DATE, SYSDATE.",
    "redshift":   "Use Amazon Redshift (PostgreSQL-compatible) syntax. Prefer DATE_TRUNC, TO_CHAR.",
}


def _dialect_hint(dialect: str) -> str:
    return _DIALECT_HINTS.get(dialect, f"Use standard SQL for {dialect}.")


SYSTEM_PROMPT_TEMPLATE = """You are a senior analytics engineer helping a product \
manager who does not know SQL. Your job is to turn their plain-English question \
into a single, correct **{dialect_label}** SELECT query.

Dialect-specific guidance: {dialect_hint}

Rules you MUST follow:
1. Respond with **nothing but** a single JSON object, exactly this shape:
   {{"sql": "<the query>", "explanation": "<one short sentence in plain English>"}}
   No prose before or after. No markdown fences. No commentary.
2. The query must be a single SELECT (or WITH ... SELECT) statement. Never INSERT / UPDATE / DELETE / DDL.
3. Only reference tables and columns that exist in the provided schema. Qualify tables with their schema when one is shown (e.g. ``public.orders``).
4. When the question is vague, pick the most reasonable interpretation and mention it in the explanation.
5. Prefer readable column aliases (e.g. ``SUM(quantity*unit_price) AS revenue``).
6. For "top N" questions, use ORDER BY ... DESC with an appropriate row limit for this dialect.
7. Default time ranges to the full available data unless the user specifies one.
8. Never fabricate columns. If the question cannot be answered with the schema, \
set "sql" to an empty string and put the reason in "explanation".
"""


@dataclass
class SQLResult:
    sql: str
    explanation: str


def _extract_json(content: str) -> dict:
    """Best-effort extraction of a JSON object from the model's reply.

    Open models sometimes wrap JSON in ```json fences or emit a short preamble
    before the object. This tolerates both.
    """
    content = (content or "").strip()

    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)```$", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # First, widest balanced-looking {...} block.
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try to trim off anything after the last closing brace and retry.
            last = candidate.rfind("}")
            if last != -1:
                try:
                    return json.loads(candidate[: last + 1])
                except json.JSONDecodeError:
                    pass

    return {"sql": "", "explanation": f"Could not parse model output: {content[:300]}"}


def _build_system_prompt(schema: Schema) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        dialect_label=schema.dialect_label,
        dialect_hint=_dialect_hint(schema.dialect),
    )


def _build_user_prompt(
    question: str,
    schema: Schema,
    history: Iterable[tuple[str, str]] = (),
    include_samples: bool = True,
) -> str:
    header = f"# Database schema ({schema.dialect_label}"
    if schema.schema_name:
        header += f", schema = {schema.schema_name}"
    header += ")"

    parts = [header, schema.to_prompt(include_samples=include_samples)]
    history = list(history)
    if history:
        parts.append("\n# Earlier questions in this conversation")
        for i, (q, s) in enumerate(history[-5:], 1):
            parts.append(f"{i}. Q: {q}\n   SQL: {s}")
    parts.append("\n# New question")
    parts.append(question.strip())
    parts.append('\nRespond with JSON only: {"sql": "...", "explanation": "..."}.')
    return "\n".join(parts)


def build_prompt_preview(
    question: str,
    schema: Schema,
    history: Iterable[tuple[str, str]] = (),
    include_samples: bool = True,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). Useful for debugging + tests."""
    return (
        _build_system_prompt(schema),
        _build_user_prompt(question, schema, history, include_samples),
    )


def _default_client(model: str) -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model=model, token=token, provider=DEFAULT_PROVIDER)


def generate_sql(
    question: str,
    schema: Schema,
    history: Iterable[tuple[str, str]] = (),
    include_samples: bool = True,
    model: str | None = None,
    client: InferenceClient | None = None,
) -> SQLResult:
    """Call the LLM and return a SQLResult.

    ``model`` defaults to ``HF_MODEL`` env var, falling back to
    ``Qwen/Qwen2.5-Coder-32B-Instruct``.
    """
    model = model or DEFAULT_MODEL
    client = client or _default_client(model)

    messages = [
        {"role": "system", "content": _build_system_prompt(schema)},
        {"role": "user",   "content": _build_user_prompt(question, schema, history, include_samples)},
    ]

    response = client.chat_completion(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content or ""
    data = _extract_json(content)
    sql = str(data.get("sql", "")).strip().rstrip(";")
    explanation = str(data.get("explanation", "")).strip()
    return SQLResult(sql=sql, explanation=explanation)
