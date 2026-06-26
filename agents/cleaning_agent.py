import pandas as pd
import numpy as np
from core.state import PipelineState

# Individual cleaning functions

def _apply_median_imputation(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    median_val = df[col].median()
    filled = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    return df, f"Column '{col}': {filled} nulls filled with median ({median_val:.4f})"


def _apply_mean_imputation(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    mean_val = df[col].mean()
    filled = df[col].isna().sum()
    df[col] = df[col].fillna(mean_val)
    return df, f"Column '{col}': {filled} nulls filled with mean ({mean_val:.4f})"


def _apply_mode_imputation(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    mode_val = df[col].mode()
    if mode_val.empty:
        return df, f"Column '{col}': mode imputation skipped (no mode found)"
    filled = df[col].isna().sum()
    df[col] = df[col].fillna(mode_val[0])
    return df, f"Column '{col}': {filled} nulls filled with mode ('{mode_val[0]}')"


def _apply_drop_rows(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    before = len(df)
    df = df.dropna(subset=[col]).reset_index(drop=True)
    dropped = before - len(df)
    return df, f"Column '{col}': {dropped} rows dropped due to null values"


def _apply_remove_duplicates(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, f"Duplicates: {removed} duplicate rows removed"


def _apply_convert_to_numeric(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, f"Column '{col}': dtype converted to numeric"


def _apply_winsorize(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    capped = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df, f"Column '{col}': {capped} outliers winsorized to [{lower:.4f}, {upper:.4f}]"


def _apply_drop_outliers(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, str]:
    col = action["column"]
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
    dropped = before - len(df)
    return df, f"Column '{col}': {dropped} outlier rows dropped"


# Action dispatcher 
ACTION_MAP = {
    "median_imputation":   _apply_median_imputation,
    "mean_imputation":     _apply_mean_imputation,
    "mode_imputation":     _apply_mode_imputation,
    "drop_rows":           _apply_drop_rows,
    "remove_duplicates":   _apply_remove_duplicates,
    "convert_to_numeric":  _apply_convert_to_numeric,
    "winsorize":           _apply_winsorize,
    "drop_outliers":       _apply_drop_outliers,
}


def run_cleaning_agent(state: PipelineState) -> PipelineState:
    """
    Executes only the approved cleaning actions from the Streamlit UI.

    Parameters
    ----------
    state : PipelineState
        Must contain state["raw_df"] and state["approved_actions"].

    Returns
    -------
    PipelineState
        Updated state with state["cleaned_df"] and state["cleaning_summary"].
    """
    # Work on a copy — never mutate the original raw data
    df = state["raw_df"].copy()
    approved_actions = state.get("approved_actions", [])
    change_log = []

    for action in approved_actions:
        action_name = action.get("suggested_action", "")
        clean_fn    = ACTION_MAP.get(action_name)

        if clean_fn is None:
            # Unknown action — skip safely and log it
            change_log.append(f"Skipped unknown action: '{action_name}'")
            continue

        try:
            df, message = clean_fn(df, action)
            change_log.append(message)
        except Exception as e:
            change_log.append(f"Error applying '{action_name}' on '{action.get('column')}': {e}")

    cleaning_summary = {
        "actions_requested": len(state.get("recommendations", [])),
        "actions_approved":  len(approved_actions),
        "actions_applied":   len(change_log),
        "final_row_count":   len(df),
        "final_col_count":   len(df.columns),
        "change_log":        change_log,
    }

    state["cleaned_df"]      = df
    state["cleaning_summary"] = cleaning_summary
    return state
