import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import seaborn as sns
from core.state import PipelineState

# All charts are saved here
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def _plot_histograms(df: pd.DataFrame) -> list:
    """Generate one histogram per numeric column."""
    chart_paths = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution — {col}", fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        path = os.path.join(CHARTS_DIR, f"hist_{col}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        chart_paths.append(path)

    return chart_paths


def _plot_boxplots(df: pd.DataFrame) -> list:
    """Generate one box plot per numeric column to visualise spread + outliers."""
    chart_paths = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(df[col].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", color="#2c4a7a"),
                   medianprops=dict(color="orange", linewidth=2))
        ax.set_title(f"Box Plot — {col}", fontsize=13)
        ax.set_xlabel(col)
        plt.tight_layout()

        path = os.path.join(CHARTS_DIR, f"box_{col}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        chart_paths.append(path)

    return chart_paths


def _plot_correlation_heatmap(df: pd.DataFrame) -> list:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []   

    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, corr_matrix.shape[0]), max(6, corr_matrix.shape[0] - 1)))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def _plot_categorical_bars(df: pd.DataFrame) -> list:
    """Generate bar charts for the top 5 most common values in categorical columns."""
    chart_paths = []
    # Limit to columns with ≤ 30 unique values to avoid unusable charts
    cat_cols = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if df[col].nunique() <= 30
    ]

    for col in cat_cols:
        value_counts = df[col].value_counts().head(10)   # Top 10 categories
        fig, ax = plt.subplots(figsize=(8, 4))
        value_counts.plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
        ax.set_title(f"Top Categories — {col}", fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path = os.path.join(CHARTS_DIR, f"bar_{col}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        chart_paths.append(path)

    return chart_paths


def run_eda_agent(state: PipelineState) -> PipelineState:
    df: pd.DataFrame = state["cleaned_df"]

    # Generate all charts
    all_chart_paths = []
    all_chart_paths.extend(_plot_histograms(df))
    all_chart_paths.extend(_plot_boxplots(df))
    all_chart_paths.extend(_plot_correlation_heatmap(df))
    all_chart_paths.extend(_plot_categorical_bars(df))

    # Summary statistics (used by insight agent) 
    numeric_df = df.select_dtypes(include=[np.number])
    summary_stats = {}
    for col in numeric_df.columns:
        summary_stats[col] = {
            "mean":   round(float(numeric_df[col].mean()), 4),
            "median": round(float(numeric_df[col].median()), 4),
            "std":    round(float(numeric_df[col].std()), 4),
            "min":    round(float(numeric_df[col].min()), 4),
            "max":    round(float(numeric_df[col].max()), 4),
            "skew":   round(float(numeric_df[col].skew()), 4),
        }

    # Correlation matrix (for insight agent) 
    correlations = {}
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().round(4)
        correlations = corr.to_dict()

    state["eda_results"] = {
        "chart_paths":   all_chart_paths,
        "summary_stats": summary_stats,
        "correlations":  correlations,
        "row_count":     len(df),
        "col_count":     len(df.columns),
    }

    return state
