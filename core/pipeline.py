"""
Wires all 6 agents into a directed graph.The Supervisor Agent is represented here as the LangGraph coordinator that
orchestrates execution and routes outputs between the specialized agents.

    Part 1 (run_pre_approval_pipeline):
        raw_df → profiling → quality → recommendations
        Stops here so Streamlit can show the recommendations to the user.

    Part 2 (run_post_approval_pipeline):
        approved_actions → cleaning → EDA → insights
        Runs after the user has approved/rejected recommendations in the UI.
"""

from langgraph.graph import StateGraph, END
from core.state import PipelineState
from agents.profiling_agent       import run_profiling_agent
from agents.quality_agent         import run_quality_agent
from agents.recommendation_agent  import run_recommendation_agent
from agents.cleaning_agent        import run_cleaning_agent
from agents.eda_agent             import run_eda_agent
from agents.insight_agent         import run_insight_agent


def _build_pre_approval_graph():
    """
    Builds the first half of the pipeline:
    supervisor orchestrates profiling → quality detection → recommendations
    """
    graph = StateGraph(PipelineState)

    graph.add_node("profiling",      run_profiling_agent)
    graph.add_node("quality",        run_quality_agent)
    graph.add_node("recommendation", run_recommendation_agent)

    graph.set_entry_point("profiling")
    graph.add_edge("profiling",      "quality")
    graph.add_edge("quality",        "recommendation")
    graph.add_edge("recommendation", END)

    return graph.compile()


def _build_post_approval_graph():
    """
    Builds the second half of the pipeline:
    supervisor orchestrates cleaning (with approved actions) → EDA → insight generation
    """
    graph = StateGraph(PipelineState)

    graph.add_node("cleaning", run_cleaning_agent)
    graph.add_node("eda",      run_eda_agent)
    graph.add_node("insights", run_insight_agent)

    graph.set_entry_point("cleaning")
    graph.add_edge("cleaning", "eda")
    graph.add_edge("eda",      "insights")
    graph.add_edge("insights", END)

    return graph.compile()


def run_pre_approval_pipeline(state: PipelineState) -> PipelineState:
    """
    Run Phase 1: Profile → Quality → Recommendations.
    Call this after the user uploads a file.
    Returns updated state with recommendations ready for human review.
    """
    pipeline = _build_pre_approval_graph()
    return pipeline.invoke(state)


def run_post_approval_pipeline(state: PipelineState) -> PipelineState:
    """
    Run Phase 2: Clean → EDA → Insights.
    Call this after the user has approved/rejected recommendations in the UI.
    Returns fully updated state with cleaned data, charts, and insights.
    """
    pipeline = _build_post_approval_graph()
    return pipeline.invoke(state)