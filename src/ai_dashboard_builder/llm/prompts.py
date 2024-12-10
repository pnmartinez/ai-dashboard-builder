"""Prompts module for the LLM Pipeline.

This module contains all the prompts used by the LLM Pipeline, organized by their purpose.
Each prompt is a function that takes relevant parameters and returns a formatted string.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd


def create_dataset_analysis_prompt(
    df: pd.DataFrame, data_summary: Dict[str, Any], kpis: Optional[List[str]] = None
) -> str:
    """Create a prompt for dataset analysis.

    Args:
        df: The dataframe to analyze
        data_summary: Dictionary containing dataset summary information
        kpis: Optional list of KPI columns to focus on

    Returns:
        Formatted prompt string
    """
    # KPI context section
    kpi_context = ""
    if kpis:
        kpi_context = f"""
🎯 KEY PERFORMANCE INDICATORS
--------------------------
The following columns have been identified as key metrics of interest:
{', '.join(kpis)}

Please pay special attention to these KPIs in your analysis, focusing on:
• Factors that influence these metrics
• Relationships between these KPIs and other variables
• Patterns and trends in these metrics
• Recommendations for improving or optimizing these KPIs
"""

    return f"""Analyze this dataset and provide a detailed report in the following format:

{kpi_context if kpis else ''}
📊 DATASET OVERVIEW
------------------
• Total Records: {len(df)}
• Total Features: {len(df.columns)}
• Time Period: [infer from temporal column if applicable]
• Dataset Purpose: [infer from content]

📋 COLUMN ANALYSIS
-----------------
{", ".join(data_summary['columns'])}

For each column:
• Type: [data type]
• Description: [what this column represents]
• Value Range/Categories: [key values or ranges]
• Quality Issues: [missing values, anomalies]
{'• Relationship to KPIs: [describe influence on KPIs]' if kpis else ''}

🔍 KEY OBSERVATIONS
-----------------
• [List 3-5 main patterns or insights]
• [Note any data quality issues]
• [Highlight interesting relationships]
{'• [Focus on KPI drivers and correlations]' if kpis else ''}

📈 STATISTICAL HIGHLIGHTS
-----------------------
• [Key statistics and distributions]
• [Notable correlations]
• [Significant patterns]
{'• [Statistical analysis of KPI relationships]' if kpis else ''}

💡 RECOMMENDATIONS
----------------
• [Suggest data cleaning steps]
• [Propose analysis approaches]
• [Recommend focus areas]
{'• [Specific recommendations for KPI optimization]' if kpis else ''}

Sample Data Preview:
{pd.DataFrame(data_summary['sample_rows']).to_string()}

Additional Information:
- Data Types: {data_summary['data_types']}
- Unique Values: {data_summary['unique_counts']}
- Null Counts: {data_summary['null_counts']}

Please provide a comprehensive analysis following this exact structure, using the section headers and emoji markers as shown."""


def create_visualization_prompt(
    column_metadata: Dict[str, Any], sample_data: str, kpis: Optional[List[str]] = None
) -> str:
    """Create a prompt for visualization suggestions.

    Args:
        column_metadata: Dictionary containing column metadata
        sample_data: String representation of sample data
        kpis: Optional list of KPI columns to focus on

    Returns:
        Formatted prompt string
    """
    example_format = """
{
    "viz_1": {
        "type": "bar",
        "x": "COLUMNNAME_FROM_DF_AS_X_VALUES",
        "y": "COLUMNNAME_FROM_DF_AS_Y_VALUES",
        "color": "COLUMNNAME_FROM_DF_TO_USE_AS_COLOR_ENCODING or None",
        "title": "e.g. Event Distribution by Day",
        "description": "e.g. Shows event frequency across days",
        "parameters": {
            "orientation": "v",
            "aggregation": "count"
        }
    }
}"""

    kpi_context = ""
    if kpis:
        kpi_context = f"""

Additionally, include visualizations for these key metrics of interest: {', '.join(kpis)}, with an emphasis on identifying trends and relationships with other variables in the dataset."""

    return f"""Given the following dataset information, suggest appropriate visualizations for an insightful dashboard in a structured JSON format. Do not use as X or Y columns with only 1 unique value.

Column Metadata:
{json.dumps(column_metadata, indent=2)}

Sample data:
{sample_data}

Return only a JSON structure intended for Plotly where each key is a visualization ID and the value contains:
1. type: The chart type (e.g., 'line', 'bar', 'scatter', 'histogram', 'box', 'heatmap')
2. x: Column(s) for x-axis
3. y: Column(s) for y-axis (if applicable)
4. color: Column for color encoding (if applicable)
5. title: Suggested title for the visualization
6. description: What insights this visualization provides
7. parameters: Additional parameters for the chart (e.g., orientation, aggregation)

{kpi_context if kpis else ''}

Example response format:
```json{example_format}```"""


def create_pattern_explanation_prompt(
    df: pd.DataFrame, pattern_description: str
) -> str:
    """Create a prompt for pattern explanation.

    Args:
        df: The dataframe containing the pattern
        pattern_description: Description of the pattern to analyze

    Returns:
        Formatted prompt string
    """
    return f"""Analyze the following pattern in the dataset:
{pattern_description}

Dataset sample:
{df.head(3).to_string()}

Please provide:
1. Possible explanations for this pattern
2. Whether this pattern is expected or anomalous
3. Potential implications or impacts
4. Recommendations for further investigation
"""


def create_analysis_summary_prompt(analysis: str, viz_specs: Dict[str, Any]) -> str:
    """Create a prompt for summarizing analysis and visualizations.

    Args:
        analysis: The dataset analysis text
        viz_specs: The visualization specifications

    Returns:
        Formatted prompt string
    """
    return f"""Based on the following dataset analysis and visualizations created, provide a concise summary of the key findings and insights.

Dataset Analysis:
{analysis}

Visualizations Created:
{json.dumps([{
    'title': viz['title'],
    'type': viz['type'],
    'description': viz['description']
} for viz in viz_specs.values()], indent=2)}

Please provide:
1. A brief overview of the dataset's purpose and content
2. 3-5 key insights discovered from the analysis
3. The most important patterns or trends shown in the visualizations
4. Any potential recommendations or next steps for further analysis

Keep the summary concise and focused on actionable insights."""
