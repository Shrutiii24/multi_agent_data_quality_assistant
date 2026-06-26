import json
import re
from core.state import PipelineState
from core.groq_client import call_groq

SYSTEM_PROMPT = """You are a senior Product Analyst writing an Executive Decision Support Report for a financial services company like American Express.

Your job is to convert dataset statistics into a structured 3-section executive report.

Respond ONLY with a valid JSON object. No text outside the JSON.

Use this exact structure:
{
  "key_findings": [
    {
      "finding": "<One crisp sentence starting with a checkmark-worthy observation. Use actual numbers from the data. Example: Customers with annual income above $80,000 generate 2.3x higher average purchase value.>"
    }
  ],
  "recommendations": [
    {
      "trigger": "<The data pattern that prompted this recommendation. Short. Example: Strong income-purchase correlation (r=0.97)>",
      "action": "<A concrete business action written in executive language. Example: Prioritize premium customer retention campaigns targeting high-income segments above $80,000 annual income.>"
    }
  ],
  "business_impact": [
    {
      "impact": "<One sentence describing a tangible business benefit. Example: Automated duplicate detection before CRM ingestion could reduce data redundancy by an estimated 3-5%.>"
    }
  ]
}

Rules:
- key_findings: 4-5 findings, always reference actual numbers from the stats
- recommendations: 4-5 recommendations, written as executive actions not technical observations
- business_impact: 4-5 impact statements focused on business value (accuracy, efficiency, revenue, risk reduction)
- Never say "correlation is high" — say what it means for the business
- Never say "missing values detected" — say what action should be taken at a process level
- Write as if presenting to a VP or Director, not a data scientist
"""


def _build_prompt(eda_results: dict, quality_report: dict) -> str:
    lines = ["Here is the data analysis summary. Generate an Executive Decision Support Report.\n"]

    lines.append(f"Dataset: {eda_results['row_count']} rows, {eda_results['col_count']} columns\n")

    lines.append("=== SUMMARY STATISTICS ===")
    for col, stats in eda_results.get("summary_stats", {}).items():
        lines.append(
            f"'{col}': mean={stats['mean']}, median={stats['median']}, "
            f"std={stats['std']}, min={stats['min']}, max={stats['max']}, skew={stats['skew']}"
        )

    # Strong correlations
    lines.append("\n=== CORRELATIONS (|r| > 0.5) ===")
    correlations = eda_results.get("correlations", {})
    reported = set()
    for col_a, row in correlations.items():
        for col_b, r_val in row.items():
            if col_a == col_b:
                continue
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in reported:
                continue
            if abs(r_val) > 0.5:
                lines.append(f"  '{col_a}' and '{col_b}': r = {r_val}")
                reported.add(pair_key)

    # Data quality context
    lines.append("\n=== DATA QUALITY ISSUES DETECTED ===")
    missing = quality_report.get("missing_values", [])
    if missing:
        cols = [f"{m['column']} ({m['missing_pct']}% missing)" for m in missing]
        lines.append(f"Missing values in: {', '.join(cols)}")

    dup = quality_report.get("duplicates", {})
    if dup.get("has_duplicates"):
        total_rows = eda_results["row_count"]
        dup_pct = round((dup["duplicate_count"] / total_rows) * 100, 1)
        lines.append(f"Duplicate rows: {dup['duplicate_count']} ({dup_pct}% of dataset)")

    outliers = quality_report.get("outliers", [])
    if outliers:
        cols = [f"{o['column']} ({o['outlier_count']} outliers)" for o in outliers]
        lines.append(f"Outliers detected in: {', '.join(cols)}")

    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "key_findings":      [{"finding": "Report generation failed. Please retry."}],
            "recommendations":   [{"trigger": "N/A", "action": "Check logs for errors."}],
            "business_impact":   [{"impact": "Unable to generate impact assessment."}],
        }


def run_insight_agent(state: PipelineState) -> PipelineState:
    eda_results    = state["eda_results"]
    quality_report = state["quality_report"]

    user_prompt  = _build_prompt(eda_results, quality_report)
    raw_response = call_groq(SYSTEM_PROMPT, user_prompt)
    report       = _parse_response(raw_response)

    state["data_insights"] = report
    return state