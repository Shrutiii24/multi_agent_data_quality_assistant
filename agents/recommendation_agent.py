import json
import re
from core.state import PipelineState
from core.groq_client import call_groq

SYSTEM_PROMPT = """You are a senior data analyst and data quality expert.
Your job is to analyse data quality issues and recommend the best cleaning strategy for each one.

Always respond with ONLY a valid JSON array. No explanation text outside the JSON.
Each element in the array must follow this exact structure:
{
  "issue_type": "<missing_value | duplicate | type_issue | outlier>",
  "column": "<column name or 'ALL' for dataset-wide issues>",
  "suggested_action": "<short action name e.g. median_imputation>",
  "rationale": "<1-2 sentence plain English explanation of why this action is best>",
  "parameters": {}
}

Allowed suggested_action values:
  For missing values  : median_imputation, mean_imputation, mode_imputation, drop_rows, constant_fill
  For duplicates      : remove_duplicates
  For type issues     : convert_to_numeric
  For outliers        : winsorize, drop_outliers, keep_as_is
"""


def _build_user_prompt(quality_report: dict, profile_report: dict) -> str:
    lines = ["Here are the data quality issues detected. Recommend the best fix for each.\n"]

    # Missing values
    for item in quality_report.get("missing_values", []):
        col   = item["column"]
        dtype = item["dtype"]
        pct   = item["missing_pct"]
        # Pull skewness info from profile (helps choose mean vs median)
        col_stats = profile_report.get("column_profiles", {}).get(col, {})
        mean   = col_stats.get("mean",   "N/A")
        median = col_stats.get("median", "N/A")
        lines.append(
            f"MISSING VALUE — Column: '{col}' | Type: {dtype} | "
            f"Missing: {pct}% | Mean: {mean} | Median: {median}"
        )

    # Duplicates
    dup = quality_report.get("duplicates", {})
    if dup.get("has_duplicates"):
        lines.append(f"DUPLICATE ROWS — Count: {dup['duplicate_count']}")

    # Type issues
    for item in quality_report.get("type_issues", []):
        lines.append(
            f"TYPE ISSUE — Column: '{item['column']}' stored as object "
            f"but {item['conversion_rate']}% of values are numeric."
        )

    # Outliers
    for item in quality_report.get("outliers", []):
        lines.append(
            f"OUTLIER — Column: '{item['column']}' | "
            f"Count: {item['outlier_count']} ({item['outlier_pct']}%) | "
            f"Fences: [{item['lower_fence']}, {item['upper_fence']}]"
        )

    return "\n".join(lines)


def _parse_llm_response(raw_response: str) -> list:
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw_response).strip().rstrip("`")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, return a fallback so the pipeline doesn't crash
        return [{"issue_type": "parse_error", "column": "N/A",
                 "suggested_action": "manual_review",
                 "rationale": "LLM response could not be parsed. Please review manually.",
                 "parameters": {}}]


def run_recommendation_agent(state: PipelineState) -> PipelineState:
    """
    Calls the Groq LLM to generate cleaning recommendations.

    Parameters
    ----------
    state : PipelineState
        Must contain state["quality_report"] and state["profile_report"].

    Returns
    -------
    PipelineState
        Updated state with state["recommendations"] populated.
    """
    quality_report = state["quality_report"]
    profile_report = state["profile_report"]

    user_prompt  = _build_user_prompt(quality_report, profile_report)
    raw_response = call_groq(SYSTEM_PROMPT, user_prompt)

    # Parse and store
    recommendations = _parse_llm_response(raw_response)
    state["recommendations"] = recommendations
    return state
