"""Pick a sensible chart for a result DataFrame.

Heuristics (kept dumb on purpose — fancy chart-picking is a rathole):
  - 1 column, numeric  -> single metric card (no chart)
  - 2 columns, first categorical + second numeric -> bar chart
  - 2 columns, first datetime + second numeric   -> line chart
  - >= 3 columns with one datetime and numerics   -> multi-line chart
  - otherwise -> no chart, just table
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            return parsed.notna().sum() >= max(1, int(0.8 * len(parsed)))
        except Exception:
            return False
    return False


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def suggest_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Return a Plotly Figure best suited to the data, or None to skip charting."""
    if df is None or df.empty:
        return None
    if len(df) < 2 and len(df.columns) > 1:
        return None

    cols = list(df.columns)
    numeric = _numeric_cols(df)

    if len(cols) == 1 and cols[0] in numeric:
        return None

    if len(cols) == 2:
        x_col, y_col = cols
        if y_col not in numeric:
            return None
        if _is_datetime_series(df[x_col]):
            plot_df = df.copy()
            plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="coerce")
            plot_df = plot_df.sort_values(x_col)
            fig = px.line(plot_df, x=x_col, y=y_col, markers=True)
        else:
            top = df.sort_values(y_col, ascending=False).head(25)
            fig = px.bar(top, x=x_col, y=y_col)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    datetime_cols = [c for c in cols if _is_datetime_series(df[c])]
    if datetime_cols and numeric:
        x_col = datetime_cols[0]
        y_cols = [c for c in numeric if c != x_col][:5]
        if not y_cols:
            return None
        plot_df = df.copy()
        plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="coerce")
        plot_df = plot_df.sort_values(x_col)
        fig = px.line(plot_df, x=x_col, y=y_cols, markers=True)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    categorical = [c for c in cols if c not in numeric]
    if len(categorical) == 1 and len(numeric) >= 1:
        x_col = categorical[0]
        y_col = numeric[0]
        top = df.sort_values(y_col, ascending=False).head(25)
        fig = px.bar(top, x=x_col, y=y_col)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    return None
