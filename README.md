# Multi-Agent Data Quality & Analytics Assistant
### From raw datasets to executive-ready insights through a coordinated multi-agent analytics workflow.

An end-to-end multi-agent AI system that automates data profiling, quality assessment, intelligent cleaning recommendations, exploratory data analysis (EDA), and business insight generation using LangGraph, Groq LLMs, and Streamlit.

## Project Overview

Data analysts spend a significant portion of their time preparing datasets before meaningful analysis can begin. Tasks such as profiling datasets, detecting missing values, handling duplicates, validating data types, identifying outliers, and performing exploratory analysis are repetitive yet essential.

This project automates that workflow using a supervisor-driven multi-agent architecture. Instead of manually performing each preprocessing step, users simply upload a CSV or Excel dataset, review LLM-generated cleaning recommendations, and receive a cleaned dataset, visual analytics, and business-ready insights through an interactive Streamlit application.

## Problem Statement

Every analytics project begins with data preparation, yet these repetitive tasks consume a large share of an analyst's time. Manual data cleaning is often inconsistent, error-prone, and delays downstream analytics.

This project addresses that challenge by building a supervisor-coordinated multi-agent pipeline that automates the entire pre-analysis workflow while keeping a human decision-maker in control of every cleaning action.

## Key Features

* Upload CSV or Excel datasets through an interactive Streamlit interface
* Supervisor Agent orchestrates specialized agents using LangGraph
* Automated dataset profiling with summary statistics and schema analysis
* Detection of missing values, duplicate records, datatype inconsistencies, invalid entries, and statistical outliers
* LLM-generated cleaning recommendations with human approval (Accept / Modify / Skip)
* Automated execution of approved preprocessing steps
* Exploratory Data Analysis including distributions, correlations, trends, and categorical analysis
* LLM-powered business insight generation using Groq
* Download cleaned dataset and review complete cleaning history

## Architecture

Dataset Upload -> Supervisor Agent (LangGraph) -> [Profiling Agent] -> Quality Agent -> Recommendation Agent -> Human Approval (Accept / Modify / Skip) -> Cleaning Agent -> EDA Agent -> Insight Generation Agent -> Executive Analytics Report

## Tech Stack

- **Programming Language:** Python
- **Agent Framework:** LangGraph
- **LLM:** Groq API (Llama 3)
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Plotly
- **Web Framework:** Streamlit
- **Architecture:** Multi-Agent System with Supervisor Agent

  ## Project Highlights

- Multi-Agent AI architecture coordinated by a Supervisor Agent
- Human-in-the-loop data cleaning workflow (Accept / Modify / Skip)
- Automated dataset profiling and quality assessment
- Intelligent cleaning recommendations powered by LLMs
- Automated exploratory data analysis (EDA)
- LLM-generated business insights from cleaned datasets
- Interactive Streamlit web application
- Modular, extensible architecture for future agents
  
