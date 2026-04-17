"""Streamlit UI for the text-to-SQL app.

Supports multiple data sources:
  - Sample e-commerce SQLite (bundled, for demos).
  - Upload a CSV or a .sqlite file.
  - Connect to a running PostgreSQL / MySQL / SQL Server / Snowflake /
    BigQuery / etc. database via a form or a SQLAlchemy URL.

The conversation keeps context so follow-up questions work.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.database import (
    DEFAULT_PORTS,
    DIALECT_LABELS,
    UnsafeSQLError,
    build_url,
    describe_engine,
    get_engine,
    introspect,
    list_schemas,
    load_csv_into_sqlite,
    run_query,
    sqlite_url,
    test_connection,
)
from src.llm import generate_sql
from src.sample_data import ensure_sample_db
from src.visualize import suggest_chart


load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SAMPLE_DB = DATA_DIR / "sample.db"
UPLOAD_DB = DATA_DIR / "uploaded.db"

EXAMPLE_QUESTIONS = [
    "What were our top 5 best-selling products last quarter?",
    "Which region generated the most revenue?",
    "Show monthly revenue trend for 2024",
    "Who are our top 10 customers by lifetime spend?",
    "What's the average order value by product category?",
    "How many new customers did we get each month?",
]


st.set_page_config(
    page_title="Ask Your Data",
    page_icon="💬",
    layout="wide",
)


# ---------------------------- session helpers ----------------------------


def _init_state() -> None:
    ss = st.session_state
    if "db_url" not in ss:
        ensure_sample_db(SAMPLE_DB)
        ss.db_url = sqlite_url(SAMPLE_DB)
        ss.db_label = "Sample e-commerce database"
    ss.setdefault("db_schema_name", None)
    ss.setdefault("include_sample_rows", True)
    ss.setdefault("turns", [])
    ss.setdefault("history_for_llm", [])
    ss.setdefault("pending_question", None)
    ss.setdefault("last_connect_error", None)


def _reset_conversation() -> None:
    st.session_state.turns = []
    st.session_state.history_for_llm = []


def _set_data_source(url: str, label: str, schema_name: str | None = None,
                     include_samples: bool = True) -> None:
    st.session_state.db_url = url
    st.session_state.db_label = label
    st.session_state.db_schema_name = schema_name
    st.session_state.include_sample_rows = include_samples
    _reset_conversation()


def _switch_to_sample() -> None:
    ensure_sample_db(SAMPLE_DB)
    _set_data_source(sqlite_url(SAMPLE_DB), "Sample e-commerce database",
                     schema_name=None, include_samples=True)


def _handle_csv_upload(uploaded) -> None:
    table = load_csv_into_sqlite(uploaded.getvalue(), uploaded.name, UPLOAD_DB)
    _set_data_source(sqlite_url(UPLOAD_DB), f"Uploaded CSV → table `{table}`",
                     include_samples=True)


def _handle_sqlite_upload(uploaded) -> None:
    UPLOAD_DB.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DB.write_bytes(uploaded.getvalue())
    _set_data_source(sqlite_url(UPLOAD_DB), f"Uploaded database: {uploaded.name}",
                     include_samples=True)


def _try_connect(url: str, label_hint: str, *, include_samples: bool) -> bool:
    """Try to connect. On success, adopt the new source. Return True/False."""
    try:
        engine = get_engine(url)
        test_connection(engine)
    except Exception as exc:  # noqa: BLE001
        st.session_state.last_connect_error = str(exc)
        return False

    label = label_hint or describe_engine(engine)
    _set_data_source(url, label, schema_name=None, include_samples=include_samples)
    st.session_state.last_connect_error = None
    engine.dispose()
    return True


# ------------------------------ sidebar UI ------------------------------


def _sidebar_connection_form() -> None:
    """The 'connect to a running database' panel."""
    with st.expander("🔌 Connect to a database", expanded=False):
        st.caption(
            "Point the app at your team's PostgreSQL, MySQL, Snowflake, "
            "BigQuery, etc. **Read-only credentials are strongly recommended.**"
        )

        tab_form, tab_url = st.tabs(["Guided form", "Connection URL"])

        with tab_form:
            supported = ["postgresql", "mysql", "mssql", "snowflake"]
            labels = [DIALECT_LABELS[k] for k in supported]
            idx = st.selectbox("Database type", range(len(supported)),
                               format_func=lambda i: labels[i])
            driver = supported[idx]

            default_port = DEFAULT_PORTS.get(driver, 0)
            c1, c2 = st.columns([3, 1])
            host = c1.text_input("Host", value="localhost", key=f"host_{driver}")
            port = c2.number_input("Port", value=default_port, min_value=0,
                                   max_value=65535, step=1, key=f"port_{driver}")
            database = st.text_input("Database", key=f"db_{driver}")
            c3, c4 = st.columns(2)
            user = c3.text_input("User", key=f"user_{driver}")
            password = c4.text_input("Password", type="password", key=f"pw_{driver}")
            include_samples = st.checkbox(
                "Include a few sample rows in the prompt (helps accuracy)",
                value=False,
                key=f"samples_{driver}",
                help=(
                    "When on, up to 3 rows per table are sent to the LLM with "
                    "the schema. Turn off for databases with sensitive data."
                ),
            )

            if st.button("Test & connect", key=f"connect_{driver}",
                         use_container_width=True):
                url = build_url(driver, host=host, database=database,
                                user=user, password=password,
                                port=int(port) if port else None)
                label = f"{DIALECT_LABELS[driver]} · {user}@{host}/{database}" if user \
                        else f"{DIALECT_LABELS[driver]} · {host}/{database}"
                if _try_connect(url, label, include_samples=include_samples):
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error(f"Could not connect: {st.session_state.last_connect_error}")

        with tab_url:
            st.caption(
                "Paste any SQLAlchemy URL, e.g. "
                "`postgresql+psycopg2://user:pass@host:5432/db` or "
                "`snowflake://user:pass@account/db/schema?warehouse=WH`."
            )
            url = st.text_area("Connection URL", height=80, key="raw_url")
            include_samples_u = st.checkbox(
                "Include a few sample rows in the prompt",
                value=False,
                key="samples_url",
            )
            if st.button("Test & connect", key="connect_url",
                         use_container_width=True):
                if not url.strip():
                    st.error("Please paste a connection URL.")
                elif _try_connect(url.strip(), label_hint="",
                                  include_samples=include_samples_u):
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error(f"Could not connect: {st.session_state.last_connect_error}")

        st.caption(
            "Drivers for PostgreSQL and MySQL ship with this app. For "
            "Snowflake / BigQuery / SQL Server / Oracle, install the "
            "corresponding driver (see README)."
        )


def _sidebar_upload() -> None:
    with st.expander("📁 Upload a file (CSV or SQLite)", expanded=False):
        csv_file = st.file_uploader("CSV file", type=["csv"], key="csv_up")
        if csv_file is not None and st.button("Load CSV", use_container_width=True):
            try:
                _handle_csv_upload(csv_file)
                st.success("Loaded. Ask a question below.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load CSV: {exc}")

        db_file = st.file_uploader("SQLite database (.db, .sqlite)",
                                   type=["db", "sqlite"], key="db_up")
        if db_file is not None and st.button("Load database", use_container_width=True):
            try:
                _handle_sqlite_upload(db_file)
                st.success("Loaded. Ask a question below.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load database: {exc}")


def _sidebar_schema_picker(engine) -> None:
    """If the DB exposes multiple schemas, let the PM pick one."""
    schemas = list_schemas(engine)
    if len(schemas) <= 1:
        return

    current = st.session_state.db_schema_name or ""
    options = [""] + schemas
    try:
        default_idx = options.index(current)
    except ValueError:
        default_idx = 0

    picked = st.selectbox(
        "Schema",
        options,
        index=default_idx,
        format_func=lambda v: v or "(default)",
        help="Limit the AI to tables in this schema. Helpful for big databases.",
    )
    if (picked or None) != st.session_state.db_schema_name:
        st.session_state.db_schema_name = picked or None
        _reset_conversation()
        st.rerun()


def _sidebar_schema_preview(engine) -> None:
    try:
        schema = introspect(
            engine,
            schema_name=st.session_state.db_schema_name,
            sample_rows=0,  # previews never include actual data
            max_tables=200,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read schema: {exc}")
        return

    if not schema.tables:
        st.info("No tables visible with the current settings.")
        return

    with st.expander(f"Schema ({len(schema.tables)} tables)", expanded=False):
        for tbl in schema.tables:
            st.markdown(f"**{tbl.qualified_name()}**")
            st.caption(
                ", ".join(
                    f"{c.name} ({c.type})" + (" pk" if c.primary_key else "")
                    for c in tbl.columns
                )
            )


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("💬 Ask Your Data")
        st.caption("Plain-English questions → SQL → answers.")

        if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
            st.warning(
                "HF_TOKEN is not set. Copy `.env.example` to `.env` and add your "
                "free Hugging Face token (huggingface.co/settings/tokens).",
                icon="⚠️",
            )

        st.subheader("Data source")
        st.write(f"**Current:** {st.session_state.db_label}")

        if st.button("Use sample e-commerce database", use_container_width=True):
            _switch_to_sample()
            st.rerun()

        _sidebar_connection_form()
        _sidebar_upload()

        # Live schema tools for the current connection.
        try:
            engine = get_engine(st.session_state.db_url)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Engine error: {exc}")
            return

        _sidebar_schema_picker(engine)

        st.checkbox(
            "Include sample rows in the prompt",
            key="include_sample_rows",
            help=(
                "Sends up to 3 rows per table to the LLM alongside the schema. "
                "Improves accuracy on categorical columns. Disable for sensitive data."
            ),
        )

        _sidebar_schema_preview(engine)

        st.subheader("Try an example")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        if st.session_state.turns:
            st.subheader("History")
            for i, turn in enumerate(reversed(st.session_state.turns)):
                if st.button(
                    turn["question"], key=f"hist_{i}",
                    use_container_width=True, help="Click to ask again",
                ):
                    st.session_state.pending_question = turn["question"]
                    st.rerun()
            if st.button("Clear history", use_container_width=True):
                _reset_conversation()
                st.rerun()


# ------------------------------- main area -------------------------------


def _render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        if turn.get("error_before_sql"):
            st.error(turn["error_before_sql"])
            return

        if turn.get("explanation"):
            st.write(turn["explanation"])

        sql = turn["sql"]
        edited = st.text_area(
            "Generated SQL (you can edit and re-run)",
            value=sql,
            key=f"sql_{index}",
            height=max(80, min(260, 24 * (sql.count(chr(10)) + 2))),
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            rerun = st.button("Run", key=f"run_{index}")
        with col2:
            st.caption("Only SELECT queries are executed.")

        run_sql = edited if rerun else sql
        if rerun or (turn.get("df") is None and not turn.get("run_error")):
            try:
                df = run_query(get_engine(st.session_state.db_url), run_sql)
                turn["df"] = df
                turn["run_error"] = None
                turn["sql"] = run_sql
            except UnsafeSQLError as exc:
                turn["run_error"] = f"Blocked: {exc}"
                turn["df"] = None
            except Exception as exc:  # noqa: BLE001
                turn["run_error"] = f"Query failed: {exc}"
                turn["df"] = None

        if turn.get("run_error"):
            st.error(turn["run_error"])
            return

        df: pd.DataFrame | None = turn.get("df")
        if df is None:
            return

        if df.empty:
            st.info("Query ran successfully but returned no rows.")
            return

        if df.shape == (1, 1):
            value = df.iat[0, 0]
            label = str(df.columns[0]).replace("_", " ").title()
            st.metric(
                label,
                f"{value:,}" if isinstance(value, (int, float)) else str(value),
            )
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(df):,} row(s) · {len(df.columns)} column(s)")
            fig = suggest_chart(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"result_{index + 1}.csv",
                mime="text/csv",
                key=f"dl_{index}",
            )


def _answer(question: str) -> None:
    try:
        engine = get_engine(st.session_state.db_url)
        schema = introspect(
            engine,
            schema_name=st.session_state.db_schema_name,
            sample_rows=3 if st.session_state.include_sample_rows else 0,
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state.turns.append(
            {"question": question, "sql": "", "explanation": "",
             "error_before_sql": f"Could not read schema: {exc}"}
        )
        return

    try:
        result = generate_sql(
            question=question,
            schema=schema,
            history=st.session_state.history_for_llm,
            include_samples=st.session_state.include_sample_rows,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=1)
        st.session_state.turns.append(
            {"question": question, "sql": "", "explanation": "",
             "error_before_sql": f"Could not generate SQL: {exc}\n{tb}"}
        )
        return

    if not result.sql:
        st.session_state.turns.append(
            {"question": question, "sql": "", "explanation": "",
             "error_before_sql": result.explanation or "I couldn't turn that into SQL."}
        )
        return

    turn = {
        "question": question,
        "sql": result.sql,
        "explanation": result.explanation,
        "df": None,
        "run_error": None,
    }
    st.session_state.turns.append(turn)
    st.session_state.history_for_llm.append((question, result.sql))


def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("Ask Your Data")
    st.caption(
        "Type a question in plain English. I'll write the SQL, run it, and show the answer."
    )

    for i, turn in enumerate(st.session_state.turns):
        _render_turn(turn, i)

    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        _answer(pending)
        st.rerun()

    question = st.chat_input("Ask a question about your data…")
    if question:
        _answer(question)
        st.rerun()


if __name__ == "__main__":
    main()
