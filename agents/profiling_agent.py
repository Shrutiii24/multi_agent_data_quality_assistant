import pandas as pd
import numpy as np
from core.state import PipelineState


def run_profiling_agent(state: PipelineState) -> PipelineState:
    """
    Profiles the raw DataFrame and stores results in state.

    Parameters
    ----------
    state : PipelineState
        Must contain state["raw_df"] (a loaded pandas DataFrame).

    Returns
    -------
    PipelineState
        Updated state with state["profile_report"] populated.
    """
    df: pd.DataFrame = state["raw_df"]

    # Dataset-level overview
    overview = {
        "total_rows":    int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names":  df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Per-column analysis
    column_profiles = {}
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isna().sum())
        missing_pct   = round((missing_count / len(df)) * 100, 2)
        unique_count  = int(col_data.nunique())

        profile = {
            "dtype":         str(col_data.dtype),
            "missing_count": missing_count,
            "missing_pct":   missing_pct,
            "unique_values": unique_count,
        }

        # Add numeric summary stats only for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            profile.update({
                "mean":   round(float(col_data.mean(skipna=True)), 4) if not col_data.dropna().empty else None,
                "median": round(float(col_data.median(skipna=True)), 4) if not col_data.dropna().empty else None,
                "std":    round(float(col_data.std(skipna=True)), 4) if not col_data.dropna().empty else None,
                "min":    round(float(col_data.min(skipna=True)), 4) if not col_data.dropna().empty else None,
                "max":    round(float(col_data.max(skipna=True)), 4) if not col_data.dropna().empty else None,
            })
        else:
            top_values = col_data.value_counts().head(5).to_dict()
            profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        column_profiles[col] = profile

    profile_report = {
        "overview":         overview,
        "column_profiles":  column_profiles,
    }

    state["profile_report"] = profile_report
    return state
