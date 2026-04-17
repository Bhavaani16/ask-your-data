"""Database helpers: connect to any SQLAlchemy-supported DB, introspect, run queries safely.

Works with SQLite, PostgreSQL, MySQL, SQL Server, Snowflake, BigQuery, etc. —
anything SQLAlchemy has a dialect for. The higher layers (LLM + UI) only see
an ``Engine`` and a ``Schema`` object, so they don't care which backend is live.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL, make_url


DEFAULT_ROW_LIMIT = 1000

_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REINDEX",
}

# Human-friendly names for dialect families. Used in the LLM prompt.
DIALECT_LABELS: dict[str, str] = {
    "sqlite":     "SQLite",
    "postgresql": "PostgreSQL",
    "mysql":      "MySQL",
    "mariadb":    "MariaDB",
    "mssql":      "Microsoft SQL Server (T-SQL)",
    "oracle":     "Oracle",
    "snowflake":  "Snowflake",
    "bigquery":   "Google BigQuery",
    "redshift":   "Amazon Redshift",
}

# Default TCP ports per driver — handy for the UI's connection form.
DEFAULT_PORTS: dict[str, int] = {
    "postgresql": 5432,
    "mysql":      3306,
    "mariadb":    3306,
    "mssql":      1433,
    "oracle":     1521,
    "redshift":   5439,
}


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)
    schema_name: Optional[str] = None

    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.name}" if self.schema_name else self.name

    def to_prompt_block(self, include_samples: bool = True) -> str:
        cols = ", ".join(
            f"{c.name} {c.type}" + (" PRIMARY KEY" if c.primary_key else "")
            for c in self.columns
        )
        block = f"TABLE {self.qualified_name()} ({cols})"
        if include_samples and self.sample_rows:
            block += "\n  -- sample rows: " + str(self.sample_rows[:3])
        return block


@dataclass
class Schema:
    tables: list[TableInfo] = field(default_factory=list)
    dialect: str = "sqlite"
    schema_name: Optional[str] = None

    def to_prompt(self, include_samples: bool = True) -> str:
        return "\n".join(t.to_prompt_block(include_samples) for t in self.tables)

    def table_names(self) -> list[str]:
        return [t.qualified_name() for t in self.tables]

    @property
    def dialect_label(self) -> str:
        return DIALECT_LABELS.get(self.dialect, self.dialect)


# ----------------------- connection / URL handling -----------------------


def sqlite_url(db_path: str | Path) -> str:
    """Build a SQLAlchemy URL for a local SQLite file."""
    return f"sqlite:///{Path(db_path).resolve()}"


def build_url(
    driver: str,
    host: str,
    database: str,
    user: str = "",
    password: str = "",
    port: Optional[int] = None,
) -> str:
    """Build a SQLAlchemy URL from form fields, with proper escaping."""
    driver = driver.strip().lower()
    if driver in {"postgres", "postgresql"}:
        drivername = "postgresql+psycopg2"
    elif driver in {"mysql", "mariadb"}:
        drivername = "mysql+pymysql"
    elif driver == "mssql":
        drivername = "mssql+pyodbc"
    elif driver == "snowflake":
        drivername = "snowflake"
    else:
        drivername = driver

    url = URL.create(
        drivername=drivername,
        username=user or None,
        password=password or None,
        host=host or None,
        port=port,
        database=database or None,
    )
    return url.render_as_string(hide_password=False)


def _read_only_connect_args(drivername: str) -> dict:
    """Best-effort hint to the driver that we only want to read.

    This is a soft safety net on top of the SELECT-only validator. If a driver
    doesn't support it, we just skip — the validator still blocks writes.
    """
    family = drivername.split("+", 1)[0]
    if family == "postgresql":
        return {"options": "-c default_transaction_read_only=on"}
    return {}


def get_engine(url: str) -> Engine:
    """Create an Engine from a full SQLAlchemy URL."""
    parsed = make_url(url)
    connect_args = _read_only_connect_args(parsed.drivername)
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def describe_engine(engine: Engine) -> str:
    """Short one-line description of what we're connected to, for the UI."""
    url = engine.url
    label = DIALECT_LABELS.get(url.get_backend_name(), url.get_backend_name())
    if url.get_backend_name() == "sqlite":
        return f"{label} · {url.database}"
    host = url.host or "localhost"
    db = url.database or ""
    user = f"{url.username}@" if url.username else ""
    return f"{label} · {user}{host}/{db}"


def test_connection(engine: Engine) -> None:
    """Open a connection and run ``SELECT 1``. Raises on failure."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def list_schemas(engine: Engine) -> list[str]:
    """Return a list of schemas the user can pick from (DB-dependent)."""
    try:
        return inspect(engine).get_schema_names() or []
    except Exception:
        return []


# ----------------------------- introspection -----------------------------


def introspect(
    engine: Engine,
    schema_name: Optional[str] = None,
    only_tables: Optional[list[str]] = None,
    sample_rows: int = 3,
    max_tables: int = 40,
) -> Schema:
    """Read table + column metadata, and optionally a few sample rows per table.

    Args:
        engine: live SQLAlchemy engine.
        schema_name: DB schema to inspect (e.g. ``public`` for Postgres).
                     ``None`` means the driver's default.
        only_tables: if given, restrict to this allow-list of table names.
        sample_rows: rows to fetch per table for the LLM prompt. Set to 0 to
                     avoid sending any actual data to the model.
        max_tables: hard cap to keep the LLM prompt small on huge schemas.
    """
    insp = inspect(engine)
    backend = engine.url.get_backend_name()
    schema = Schema(dialect=backend, schema_name=schema_name)

    try:
        table_names = insp.get_table_names(schema=schema_name)
    except Exception:
        table_names = insp.get_table_names()

    if only_tables:
        allowed = {t.lower() for t in only_tables}
        table_names = [t for t in table_names if t.lower() in allowed]

    table_names = table_names[:max_tables]

    for table_name in table_names:
        cols: list[ColumnInfo] = []
        try:
            pk_cols = set(
                insp.get_pk_constraint(table_name, schema=schema_name)
                .get("constrained_columns") or []
            )
        except Exception:
            pk_cols = set()

        try:
            col_defs = insp.get_columns(table_name, schema=schema_name)
        except Exception:
            col_defs = []

        for col in col_defs:
            cols.append(
                ColumnInfo(
                    name=col["name"],
                    type=str(col["type"]),
                    nullable=bool(col.get("nullable", True)),
                    primary_key=col["name"] in pk_cols,
                )
            )

        samples: list[dict] = []
        if sample_rows > 0:
            try:
                qualified = (
                    f'"{schema_name}"."{table_name}"' if schema_name else f'"{table_name}"'
                )
                with engine.connect() as conn:
                    result = conn.execute(
                        text(f"SELECT * FROM {qualified} LIMIT :n"),
                        {"n": sample_rows},
                    )
                    samples = [dict(r._mapping) for r in result]
            except Exception:
                samples = []

        schema.tables.append(
            TableInfo(
                name=table_name,
                columns=cols,
                sample_rows=samples,
                schema_name=schema_name,
            )
        )
    return schema


# --------------------------- safe query execution --------------------------


class UnsafeSQLError(ValueError):
    """Raised when the generated SQL tries to do something other than SELECT."""


def is_safe_select(sql: str) -> tuple[bool, Optional[str]]:
    """Return (ok, reason_if_not_ok).

    Rules:
      - Exactly one statement.
      - Must start with SELECT or WITH (a CTE ending in SELECT).
      - Must not contain any forbidden keyword.
    """
    if not sql or not sql.strip():
        return False, "Empty query."

    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) > 1:
        return False, "Only one statement is allowed."

    stmt = statements[0]
    parsed = sqlparse.parse(stmt)[0]
    first_token = next(
        (t for t in parsed.tokens
         if not t.is_whitespace and t.ttype is not sqlparse.tokens.Comment),
        None,
    )
    if first_token is None:
        return False, "Could not parse query."

    head = first_token.value.upper()
    if head not in {"SELECT", "WITH"}:
        return False, f"Only SELECT queries are allowed (got {head})."

    upper = stmt.upper()
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"Query contains forbidden keyword: {kw}."

    return True, None


def ensure_row_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Append a LIMIT if the query doesn't already constrain its output.

    Uses the SQLite/Postgres/MySQL style (``LIMIT N``). For SQL Server the
    model is told to use ``TOP`` directly in the prompt, so this usually won't
    be triggered; if it is, SQL Server will raise and the PM sees a clear error.
    """
    stripped = sql.strip().rstrip(";").rstrip()
    if re.search(r"\bLIMIT\b\s+\d+", stripped, flags=re.IGNORECASE):
        return stripped
    if re.search(r"\bTOP\b\s+\d+", stripped, flags=re.IGNORECASE):
        return stripped
    if re.search(r"\bFETCH\s+(FIRST|NEXT)\b", stripped, flags=re.IGNORECASE):
        return stripped
    return f"{stripped}\nLIMIT {limit}"


def run_query(engine: Engine, sql: str, row_limit: int = DEFAULT_ROW_LIMIT) -> pd.DataFrame:
    """Validate + execute a SELECT query and return a DataFrame."""
    ok, reason = is_safe_select(sql)
    if not ok:
        raise UnsafeSQLError(reason or "Unsafe SQL.")
    bounded = ensure_row_limit(sql, row_limit)
    with engine.connect() as conn:
        return pd.read_sql_query(text(bounded), conn)


# --------------------------- CSV upload → SQLite ---------------------------


_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_table_name(name: str) -> str:
    stem = Path(name).stem.lower()
    cleaned = _IDENT_RE.sub("_", stem).strip("_")
    if not cleaned:
        cleaned = "uploaded"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def load_csv_into_sqlite(
    csv_bytes: bytes,
    source_name: str,
    db_path: Path,
    table_name: Optional[str] = None,
) -> str:
    """Load a CSV file into a fresh SQLite database and return the table name."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    df = pd.read_csv(io.BytesIO(csv_bytes))
    table = _sanitize_table_name(table_name or source_name)
    engine = create_engine(f"sqlite:///{db_path.resolve()}")
    df.to_sql(table, engine, index=False)
    engine.dispose()
    return table
