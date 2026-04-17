# Ask Your Data — Text-to-SQL for Product Managers

A friendly web app that turns plain-English questions into SQL queries, runs them against your data, and shows the answer as a table and chart. **No SQL knowledge required.**

Built for product managers, analysts, and anyone who wants answers from data without bugging an engineer.

## What it does

- Type a question in plain English ("How many orders did we get last month by region?")
- See the SQL that was generated (and edit it if you want)
- Get the results as a clean table
- Automatic chart suggestion (bar / line) when it makes sense
- Ask follow-ups — the app remembers context ("now break that down by category")
- Query history in the sidebar — one click to re-run
- Ships with a sample e-commerce database so you can try it in 30 seconds
- Upload your own CSV or SQLite file and query it the same way

## Quick start

### 1. Install

```bash
cd text2sql
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Hugging Face token

```bash
cp .env.example .env
# then open .env and paste your token
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a "Read" token is enough. The app uses Hugging Face's serverless Inference API, so no GPU or local model download is needed.

### 3. Run

```bash
streamlit run app.py
```

Your browser opens to `http://localhost:8501`. That's it.

## Example questions to try on the sample data

- *What were our top 5 best-selling products last quarter?*
- *Which region generated the most revenue?*
- *Show monthly revenue trend for 2024*
- *Who are our top 10 customers by lifetime spend?*
- *What's the average order value by product category?*
- *How many new customers did we get each month?*

## Using your own data

You have three options — pick whichever fits your situation.

### Option A — Connect to a running database (PostgreSQL, MySQL, Snowflake, BigQuery, …)

Click **"🔌 Connect to a database"** in the sidebar. You get two ways in:

- **Guided form** — pick the database type from a dropdown and fill in host, port, user, password, and database name. Click **Test & connect**.
- **Connection URL** — paste any [SQLAlchemy connection URL](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls). Examples:

  ```
  postgresql+psycopg2://readonly_user:pass@db.company.com:5432/analytics
  mysql+pymysql://readonly_user:pass@db.company.com:3306/app
  snowflake://user:pass@account_id/db_name/schema?warehouse=WH&role=READONLY
  bigquery://my-project/my_dataset
  mssql+pyodbc://user:pass@server:1433/db?driver=ODBC+Driver+18+for+SQL+Server
  ```

**Drivers.** PostgreSQL (`psycopg2-binary`) and MySQL (`pymysql`) ship with the app. For other systems install the corresponding driver into the same venv:

```bash
pip install "snowflake-sqlalchemy>=1.6.1"       # Snowflake
pip install "sqlalchemy-bigquery>=1.11.0"       # BigQuery (plus: gcloud auth application-default login)
pip install pyodbc                               # SQL Server (requires the ODBC driver installed on your OS)
pip install cx_Oracle                            # Oracle (requires Oracle Instant Client)
```

**Big schemas?** If your database has many schemas (e.g. Postgres `public`, `raw`, `analytics`), a **Schema** selector appears in the sidebar after you connect — pick one to scope the AI to just those tables.

**Sensitive data?** Next to the schema picker is a **"Include sample rows in the prompt"** toggle. It's off by default for live connections. When off, only table/column names and types are sent to the LLM — no actual data rows.

### Option B — Upload a CSV

Open **"📁 Upload a file"** in the sidebar → choose a `.csv`. It becomes a table you can query immediately.

### Option C — Upload a SQLite database

Upload any `.db` / `.sqlite` file and all its tables become queryable.

## Safety

- Only `SELECT` queries are allowed — the app rejects `INSERT`, `UPDATE`, `DELETE`, `DROP`, and any other write / DDL statement before it ever reaches your database.
- For PostgreSQL connections, the app also opens the session with `default_transaction_read_only = on` as a defense in depth.
- Results are capped (default 1,000 rows) to keep the UI snappy.
- Generated SQL is always shown so you can verify it before trusting the answer.
- **Still strongly recommended:** connect with a read-only database user. The app's own guards are belt-and-suspenders — least privilege at the DB layer is the best protection.

## Project layout

```
text2sql/
├── app.py                 Streamlit entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── database.py        schema introspection, safe query execution, CSV upload
│   ├── llm.py             prompt construction + OpenAI call + SQL extraction
│   ├── sample_data.py     builds the demo e-commerce DB
│   └── visualize.py       picks a sensible chart for the result set
└── data/
    └── sample.db          auto-created on first run
```

## Tech

- **Streamlit** for the UI
- **Hugging Face Inference** for the language model (default `Qwen/Qwen2.5-Coder-32B-Instruct` — one of the strongest open models for SQL). Swap via the `HF_MODEL` env var. Other good picks:
  - `meta-llama/Meta-Llama-3.1-70B-Instruct`
  - `mistralai/Mistral-Nemo-Instruct-2407`
  - `defog/llama-3-sqlcoder-8b` (SQL-specialised, smaller/faster)
- **SQLAlchemy** for the data layer — works with SQLite, PostgreSQL, MySQL, SQL Server, Snowflake, BigQuery, Oracle, Redshift, and anything else SQLAlchemy has a dialect for
- **Plotly** for charts
- **sqlparse** to sanity-check generated SQL

### Changing model or provider

Add to `.env`:

```bash
HF_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
# Optional: pin a specific inference provider (hf-inference, together, fireworks-ai, …)
HF_PROVIDER=hf-inference
```

If a model isn't immediately available on the free tier, Hugging Face will tell you in the error message — try a different model from the [warm models list](https://huggingface.co/models?inference=warm) or pin a different provider.
