"""
Defines the PipelineState TypedDict — the shared data structure
that flows through every node in the LangGraph pipeline. Since LangGraph requires state to be a typed dictionary so it can track
what each node reads and writes. 
"""

from typing import Any, Optional
from typing_extensions import TypedDict
import pandas as pd

class PipelineState(TypedDict):
    raw_df: Optional[pd.DataFrame]     
    profile_report: Optional[dict]     
    quality_report: Optional[dict]           
    recommendations: Optional[list]  
    approved_actions: Optional[list]    
    cleaned_df: Optional[pd.DataFrame]       
    cleaning_summary: Optional[dict]   
    eda_results: Optional[dict]    
    data_insights: Optional[list]                 
