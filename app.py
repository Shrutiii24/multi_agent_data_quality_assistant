import streamlit as st
import pandas as pd
import os
from core.state    import PipelineState
from core.pipeline import run_pre_approval_pipeline, run_post_approval_pipeline

st.set_page_config(
    page_title="Agentic Data Assistant",
    layout="wide",
)

def init_session_state():
    defaults = {
        "pipeline_state":    None,  
        "phase1_done":       False,  
        "phase2_done":       False, 
        "approved_actions":  None,  
        "active_page":       "upload",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def render_sidebar():
    st.sidebar.title("Agentic Data Assistant")
    st.sidebar.markdown("---")

    pages = {
        "upload":          "Upload Dataset",
        "profile":         "Data Profile",
        "quality":         "Quality Issues",
        "recommendations": "Recommendations",
        "cleaning":        "Cleaning Summary",
        "eda":             "EDA Charts",
        "insights":        "Executive Analytics Report",
    }

    for page_key, page_label in pages.items():
        # Disable pages that aren't reachable yet
        is_disabled = (
            (page_key in ["profile", "quality", "recommendations"] and not st.session_state.phase1_done) or
            (page_key in ["cleaning", "eda", "insights"] and not st.session_state.phase2_done)
        )
        if not is_disabled:
            if st.sidebar.button(page_label, use_container_width=True):
                st.session_state.active_page = page_key
        else:
            st.sidebar.button(page_label, use_container_width=True, disabled=True)

render_sidebar()

# UPLOAD

def page_upload():
    st.title("📁 Upload Your Dataset")
    st.markdown(
        "Upload a CSV or Excel file. The pipeline will automatically profile it, "
        "detect quality issues, and suggest cleaning actions for your review."
    )

    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # Load the file into a DataFrame
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        st.success(f"File loaded: **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Analysis Pipeline", type="primary", use_container_width=True):
            with st.spinner("Supervisor Agent: orchestrating Profiling → Quality Detection → Recommendations..."):
                # Build initial state and run Phase 1
                initial_state: PipelineState = {
                    "raw_df":           df,
                    "profile_report":   None,
                    "quality_report":   None,
                    "recommendations":  None,
                    "approved_actions": None,
                    "cleaned_df":       None,
                    "cleaning_summary": None,
                    "eda_results":      None,
                    "insights":         None,
                }
                result_state = run_pre_approval_pipeline(initial_state)

            st.session_state.pipeline_state = result_state
            st.session_state.phase1_done    = True
            st.session_state.active_page    = "profile"
            st.rerun()

# DATA PROFILE

def page_profile():
    st.title("Data Profile Report")
    report = st.session_state.pipeline_state["profile_report"]
    overview = report["overview"]
    col_profiles = report["column_profiles"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",       overview["total_rows"])
    c2.metric("Total Columns",    overview["total_columns"])
    c3.metric("Duplicate Rows",   overview["duplicate_rows"])
    c4.metric("Columns with Nulls",
              sum(1 for v in col_profiles.values() if v["missing_count"] > 0))

    st.markdown("---")
    st.subheader("Column-Level Profile")

    # Build a summary table
    rows = []
    for col, info in col_profiles.items():
        row = {
            "Column":         col,
            "Type":           info["dtype"],
            "Missing Count":  info["missing_count"],
            "Missing %":      f"{info['missing_pct']}%",
            "Unique Values":  info["unique_values"],
        }
        # Add numeric stats if present
        if "mean" in info:
            row["Mean"]   = info.get("mean",   "—")
            row["Median"] = info.get("median", "—")
            row["Std"]    = info.get("std",    "—")
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("➡️ View Quality Issues"):
        st.session_state.active_page = "quality"
        st.rerun()

# QUALITY ISSUES

def page_quality():
    st.title("🔍 Data Quality Issues")
    report = st.session_state.pipeline_state["quality_report"]

    st.subheader("Missing Values")
    missing = report["missing_values"]
    if missing:
        st.dataframe(pd.DataFrame(missing), use_container_width=True)
    else:
        st.success("No missing values detected.")

    st.subheader("Duplicate Rows")
    dup = report["duplicates"]
    if dup["has_duplicates"]:
        st.warning(f"⚠️ {dup['duplicate_count']} duplicate rows found.")
    else:
        st.success("No duplicate rows detected.")

    st.subheader("Data Type Issues")
    type_issues = report["type_issues"]
    if type_issues:
        st.dataframe(pd.DataFrame(type_issues), use_container_width=True)
    else:
        st.success("No data type issues detected.")

    st.subheader("Outliers (IQR Method)")
    outliers = report["outliers"]
    if outliers:
        st.dataframe(pd.DataFrame(outliers), use_container_width=True)
    else:
        st.success("No significant outliers detected.")

    if st.button("➡️ View Recommendations"):
        st.session_state.active_page = "recommendations"
        st.rerun()

# HUMAN-IN-THE-LOOP RECOMMENDATIONS

def page_recommendations():
    st.title("Cleaning Recommendations")
    st.markdown(
        """
        The recommendations below are based on the data quality issues found and can be reviewed before proceeding.
        **Review each recommendation and choose an action before proceeding.**

        > **Accept** — Apply this fix exactly as suggested
        > **Modify** — Change the action or parameters before applying
        > **Skip** — Do not apply this fix
        """
    )

    recommendations = st.session_state.pipeline_state["recommendations"]

    if not recommendations:
        st.info("No recommendations generated. Your dataset may already be clean.")
        if st.button("➡️ Proceed to Cleaning"):
            st.session_state.pipeline_state["approved_actions"] = []
            _run_phase2()
        return
    
    user_choices = {}     

    for i, rec in enumerate(recommendations):
        with st.expander(
            f"{'🔴' if rec['issue_type'] == 'outlier' else '🟡'} "
            f"[{rec['issue_type'].upper()}] Column: `{rec['column']}` — "
            f"Suggested: `{rec['suggested_action']}`",
            expanded=True
        ):
            # Show the recommendation rationale as plain guidance
            st.markdown(f"**Reasoning:** {rec['rationale']}")

            col1, col2 = st.columns([2, 1])

            with col1:
                decision = st.radio(
                    "Your decision:",
                    options=["Accept", "Modify", "Skip"],
                    key=f"decision_{i}",
                    horizontal=True,
                )

            modified_action = rec["suggested_action"]
            if decision == "Modify":
                with col2:
                    action_options = _get_action_options_for_issue(rec["issue_type"])
                    modified_action = st.selectbox(
                        "Choose alternative action:",
                        options=action_options,
                        key=f"modified_action_{i}",
                    )

            user_choices[i] = {
                "decision":        decision,
                "original_rec":    rec,
                "modified_action": modified_action,
            }

    st.markdown("---")

    # Confirm and run Phase 2 
    if st.button("Confirm Choices & Run Cleaning", type="primary", use_container_width=True):
        approved_actions = []
        for i, choice in user_choices.items():
            if choice["decision"] == "Skip":
                continue  
            action = dict(choice["original_rec"])
            action["suggested_action"] = choice["modified_action"]
            approved_actions.append(action)

        # Write approved actions into pipeline state
        st.session_state.pipeline_state["approved_actions"] = approved_actions
        st.session_state.approved_actions = approved_actions

        _run_phase2()


def _get_action_options_for_issue(issue_type: str) -> list:
    """Return relevant alternative actions based on issue type."""
    options = {
        "missing_value": ["median_imputation", "mean_imputation", "mode_imputation", "drop_rows", "constant_fill"],
        "duplicate":     ["remove_duplicates"],
        "type_issue":    ["convert_to_numeric"],
        "outlier":       ["winsorize", "drop_outliers", "keep_as_is"],
    }
    return options.get(issue_type, ["manual_review"])


def _run_phase2():
    """Run Phase 2 pipeline and navigate to the cleaning summary page."""
    with st.spinner("Supervisor Agent: orchestrating Cleaning → EDA → Insights..."):
        updated_state = run_post_approval_pipeline(
            st.session_state.pipeline_state
        )
    st.session_state.pipeline_state = updated_state
    st.session_state.phase2_done    = True
    st.session_state.active_page    = "cleaning"
    st.rerun()

# CLEANING SUMMARY

def page_cleaning():
    st.title("Cleaning Summary")
    summary = st.session_state.pipeline_state["cleaning_summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actions Requested", summary["actions_requested"])
    c2.metric("Actions Approved",  summary["actions_approved"])
    c3.metric("Final Row Count",   summary["final_row_count"])
    c4.metric("Final Col Count",   summary["final_col_count"])

    st.markdown("---")
    st.subheader("Change Log")
    for entry in summary["change_log"]:
        st.markdown(f"- {entry}")

    st.markdown("---")
    st.subheader("Preview — Cleaned Dataset")
    cleaned_df = st.session_state.pipeline_state["cleaned_df"]
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("➡️ View EDA Charts", use_container_width=True):
            st.session_state.active_page = "eda"
            st.rerun()

# EDA CHARTS

def page_eda():
    st.title("Exploratory Data Analysis")
    eda_results = st.session_state.pipeline_state["eda_results"]
    chart_paths = eda_results.get("chart_paths", [])

    if not chart_paths:
        st.info("No charts were generated. Dataset may have no numeric/categorical columns.")
        return

    st.subheader("Summary Statistics")
    stats_rows = []
    for col, stats in eda_results.get("summary_stats", {}).items():
        stats_rows.append({"Column": col, **stats})
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Charts")

    # Display charts in a 2-column grid
    cols = st.columns(2)
    for idx, path in enumerate(chart_paths):
        if os.path.exists(path):
            with cols[idx % 2]:
                st.image(path, use_column_width=True)

    if st.button("➡️ View Business Insights"):
        st.session_state.active_page = "insights"
        st.rerun()


# INSIGHTS

def page_insights():
    st.title("Excecutive Business Insights")
    st.markdown("The assistant has analysed your cleaned dataset and generated key findings, recommendations, and potential business impact.")

    st.markdown("---")
 
    report = st.session_state.pipeline_state["data_insights"]
 
    st.subheader("Key Findings")
    st.caption("What the data reveals")
    st.markdown("")
 
    key_findings = report.get("key_findings", [])
    for item in key_findings:
        with st.container(border=True):
            st.markdown(f"✅ &nbsp; {item.get('finding', '')}")
 
    st.markdown("---")
 
    st.subheader("Business Recommendations")
    st.caption("What to do about it")
    st.markdown("")
 
    recommendations = report.get("recommendations", [])
    for item in recommendations:
        with st.container(border=True):
            st.markdown(f"**Data Signal:** `{item.get('trigger', '')}`")
            st.markdown(f"**→ Recommended Action:** {item.get('action', '')}")
 
    st.markdown("---")
 
    st.subheader("Potential Business Impact")
    st.caption("Why it matters to the organisation")
    st.markdown("")
 
    business_impact = report.get("business_impact", [])
    for item in business_impact:
        with st.container(border=True):
            st.markdown(f"🔷 &nbsp; {item.get('impact', '')}")
 
    st.markdown("---")
 
    report_text = _build_report_text(report)
    st.download_button(
        "Download Report as .txt",
        data=report_text,
        file_name="executive_decision_support_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
 
    st.markdown("")
    if st.button("Start Over with a New Dataset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
 
 
def _build_report_text(report: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("   EXECUTIVE DECISION SUPPORT REPORT")
    lines.append("   Generated by Agentic Data Quality Assistant")
    lines.append("=" * 60)
    lines.append("")
 
    lines.append("SECTION 1 — KEY FINDINGS")
    lines.append("-" * 40)
    for item in report.get("key_findings", []):
        lines.append(f"  ✓ {item.get('finding', '')}")
    lines.append("")
 
    lines.append("SECTION 2 — BUSINESS RECOMMENDATIONS")
    lines.append("-" * 40)
    for item in report.get("recommendations", []):
        lines.append(f"  Data Signal : {item.get('trigger', '')}")
        lines.append(f"  Action      : {item.get('action', '')}")
        lines.append("")
 
    lines.append("SECTION 3 — POTENTIAL BUSINESS IMPACT")
    lines.append("-" * 40)
    for item in report.get("business_impact", []):
        lines.append(f"  • {item.get('impact', '')}")
    lines.append("")
 
    lines.append("=" * 60)
    return "\n".join(lines)

# ROUTER 

PAGE_MAP = {
    "upload":          page_upload,
    "profile":         page_profile,
    "quality":         page_quality,
    "recommendations": page_recommendations,
    "cleaning":        page_cleaning,
    "eda":             page_eda,
    "insights":        page_insights,
}

current_page = st.session_state.get("active_page", "upload")
PAGE_MAP.get(current_page, page_upload)()