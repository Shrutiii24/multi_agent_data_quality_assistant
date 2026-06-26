import pandas as pd
import numpy as np
from core.state import PipelineState


def _detect_missing_values(df: pd.DataFrame) -> list:
    """Return a list of columns that have at least one missing value."""
    issues = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            issues.append({
                "column":        col,
                "dtype":         str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct":   round((missing_count / len(df)) * 100, 2),
            })
    return issues


def _detect_duplicates(df: pd.DataFrame) -> dict:
    """Return duplicate row count and a sample of duplicate indices."""
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_count": dup_count,
        "has_duplicates":  dup_count > 0,
    }


def _detect_type_issues(df: pd.DataFrame) -> list:
    """
    Detect columns stored as object/string that likely should be numeric.
    Strategy: try converting to numeric; if > 80% succeed, flag as type issue.
    """
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_original  = df[col].notna().sum()
        successfully_converted = converted.notna().sum()

        if non_null_original > 0:
            conversion_rate = successfully_converted / non_null_original
            if conversion_rate > 0.8:   # 80% of values look numeric
                issues.append({
                    "column":          col,
                    "current_dtype":   "object",
                    "suggested_dtype": "numeric",
                    "conversion_rate": round(conversion_rate * 100, 2),
                })
    return issues


def _detect_outliers_iqr(df: pd.DataFrame) -> list:
    issues = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        Q1  = col_data.quantile(0.25)
        Q3  = col_data.quantile(0.75)
        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        outlier_mask  = (col_data < lower_fence) | (col_data > upper_fence)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            issues.append({
                "column":        col,
                "outlier_count": outlier_count,
                "outlier_pct":   round((outlier_count / len(col_data)) * 100, 2),
                "lower_fence":   round(float(lower_fence), 4),
                "upper_fence":   round(float(upper_fence), 4),
            })
    return issues


def run_quality_agent(state: PipelineState) -> PipelineState:
    """
    Runs all quality checks and stores results in state.

    Parameters
    ----------
    state : PipelineState
        Must contain state["raw_df"].

    Returns
    -------
    PipelineState
        Updated state with state["quality_report"] populated.
    """
    df: pd.DataFrame = state["raw_df"]

    quality_report = {
        "missing_values":  _detect_missing_values(df),
        "duplicates":      _detect_duplicates(df),
        "type_issues":     _detect_type_issues(df),
        "outliers":        _detect_outliers_iqr(df),
    }

    state["quality_report"] = quality_report
    return state
