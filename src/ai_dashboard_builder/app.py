"""AI Dashboard Builder - A Dash application for automated dashboard creation using LLMs.

This module serves as the main entry point for the AI Dashboard Builder application.
It handles the web interface, data processing, and visualization generation using
various LLM providers.
"""

# --- 1. IMPORTS ---
# Standard library imports
import os

from dotenv import load_dotenv

from ai_dashboard_builder.utils.paths import get_root_path

# Load environment variables from the project root
load_dotenv(os.path.join(get_root_path(), ".env"))

# Unset any dummy API keys
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]:
    if os.getenv(key) == "dummy_key":
        os.environ.pop(key, None)

import base64
import glob
import io
import json
import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager

# Local imports
from ai_dashboard_builder.dashboard_builder import DashboardBuilder
from ai_dashboard_builder.llm.llm_pipeline import LLMPipeline

# --- 2. CONSTANTS ---
# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app/src

# Color Palette - Salmon theme
COLORS = {
    "background": "#FFF5F5",
    "card": "#FFF0F0",
    "divider": "#FFD7D7",
    "primary": "#FF9999",
    "secondary": "#FF7777",
    "warning": "#FFB366",
    "error": "#FF6B6B",
    "text_primary": "#4A4A4A",
    "text_secondary": "#717171",
    "highlight": "#FF8585",
    "info": "#85A3FF",
}

# Data preview limits
MAX_PREVIEW_ROWS = 1000
MAX_PREVIEW_COLS = 100

# --- 3. CONFIGURATION ---
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Add environment variable logging for debugging
logger.info("Environment variables loaded:")
logger.info(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
logger.info(
    f"ANTHROPIC_API_KEY: {'set' if os.getenv('ANTHROPIC_API_KEY') else 'not set'}"
)
logger.info(f"GROQ_API_KEY: {'set' if os.getenv('GROQ_API_KEY') else 'not set'}")
logger.info(f"OLLAMA_HOST: {os.getenv('OLLAMA_HOST', 'not set')}")

# Setup diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# --- 4. APP INITIALIZATION ---
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    ],
    suppress_callback_exceptions=False,
    update_title=None,
    long_callback_manager=long_callback_manager,
    title="AI Dashboard Builder",
)


# --- 5. UTILITY FUNCTIONS ---
def get_api_key_from_env_file(model_name: str) -> str:
    """Get the appropriate API key based on the model name from environment variables."""
    key_mapping = {
        "gpt": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "mixtral": "GROQ_API_KEY",
        "groq": "GROQ_API_KEY",
        "llama": "GROQ_API_KEY",
        "gemma": "GROQ_API_KEY",
    }

    for model_type, env_key in key_mapping.items():
        if model_type in model_name.lower():
            return os.getenv(env_key, "")
    return ""


def smart_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently convert string columns to numeric types where possible."""

    def clean_numeric_string(s: Any) -> Any:
        if pd.isna(s):
            return s
        if not isinstance(s, str):
            return s

        s = str(s).strip()
        s = re.sub(r"[($€£¥,)]", "", s)

        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        if s.endswith("%"):
            try:
                return float(s.rstrip("%")) / 100
            except Exception:
                return s

        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}
        if s and s[-1].upper() in multipliers:
            try:
                return float(s[:-1]) * multipliers[s[-1].upper()]
            except Exception:
                return s

        return s

    def try_numeric_conversion(series: pd.Series) -> pd.Series:
        cleaned = series.map(clean_numeric_string)
        try:
            numeric = pd.to_numeric(cleaned, errors="coerce")
            na_ratio = numeric.isna().sum() / len(numeric)
            if na_ratio < 0.3:
                return numeric
        except Exception:
            pass
        return series

    df_converted = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            continue

        try:
            pd.to_datetime(df[col], errors="raise")
            continue
        except Exception:
            pass

        df_converted[col] = try_numeric_conversion(df[col])

    return df_converted


def apply_filters(df: pd.DataFrame, filter_state: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    logger.info(f"Applying filters with state: {filter_state}")
    
    if not filter_state:
        logger.info("No filter state provided, returning original dataframe")
        return df

    filtered_df = df.copy()
    original_len = len(filtered_df)
    logger.info(f"Original dataframe length: {original_len}")

    temporal_filter = filter_state.get("temporal", {})
    if temporal_filter.get("start_date") and temporal_filter.get("end_date"):
        logger.info(f"Applying temporal filter: {temporal_filter}")
        for col in df.columns:
            try:
                filtered_df[col] = pd.to_datetime(filtered_df[col])
                filtered_df = filtered_df[
                    (filtered_df[col] >= temporal_filter["start_date"])
                    & (filtered_df[col] <= temporal_filter["end_date"])
                ]
                logger.info(f"Applied temporal filter on column {col}. Rows after filter: {len(filtered_df)}")
                break
            except Exception as e:
                logger.debug(f"Column {col} is not temporal: {str(e)}")
                continue

    categorical_filters = filter_state.get("categorical", {})
    for col, values in categorical_filters.items():
        if values:
            logger.info(f"Applying categorical filter on {col} with values: {values}")
            before_len = len(filtered_df)
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
            logger.info(f"Categorical filter on {col} reduced rows from {before_len} to {len(filtered_df)}")

    final_len = len(filtered_df)
    logger.info(f"Final filtered dataframe length: {final_len} (reduced by {original_len - final_len} rows)")
    return filtered_df


# --- 6. LAYOUT ---
app.layout = html.Div(
    [
        # Data stores
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="viz-state", storage_type="memory"),
        dcc.Store(id="dashboard-rendered", storage_type="memory"),
        dcc.Store(id="filter-state", storage_type="memory"),
        dcc.Store(id="selected-figure-store", storage_type="memory"),
        # CSS styles as dictionaries
        dcc.Store(
            id="chart-container-hover-styles",
            data={"opacity": 1, "backgroundColor": "#f8f9fa"},
        ),
        # Main container
        dbc.Container(
            fluid=True,
            children=[
                # Header
                html.A(
                    [
                        html.H1(
                            "AI Dashboard Builder",
                            style={
                                "textAlign": "center",
                                "color": COLORS["primary"],
                                "marginBottom": "0.5rem",
                                "paddingTop": "1rem",
                                "textDecoration": "none",
                            },
                        ),
                        html.H5(
                            "Throw your data, let AI build a dashboard",
                            style={
                                "textAlign": "center",
                                "color": COLORS["text_secondary"],
                                "marginBottom": "2rem",
                                "fontWeight": "lighter",
                                "fontStyle": "italic",
                            },
                        ),
                    ],
                    href="/",
                    style={"textDecoration": "none"},
                ),
                # Controls Section
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        # LLM Provider Column
                                        dbc.Col(
                                            [
                                                html.H5(
                                                    "LLM Provider", className="mb-2"
                                                ),
                                                dbc.RadioItems(
                                                    id="llm-provider",
                                                    options=[
                                                        {
                                                            "label": html.Span([
                                                                "Local ",
                                                                html.A(
                                                                    "(Ollama)",
                                                                    href="https://ollama.com/download",
                                                                    target="_blank",
                                                                ),
                                                            ]),
                                                            "value": "local",
                                                        },
                                                        {
                                                            "label": "External API",
                                                            "value": "external",
                                                        },
                                                    ],
                                                    value="local",
                                                    className="mb-2",
                                                    inline=True,
                                                ),
                                                dbc.Collapse(
                                                    dbc.Input(
                                                        id="api-key-input",
                                                        type="password",
                                                        placeholder="Enter API Key",
                                                        className="mb-2",
                                                    ),
                                                    id="api-key-collapse",
                                                    is_open=False,
                                                ),
                                                dbc.Collapse(
                                                    dbc.Select(
                                                        id="model-selection",
                                                        options=[
                                                            {
                                                                "label": "GPT-4o-mini",
                                                                "value": "gpt-4o-mini",
                                                            },
                                                            {
                                                                "label": "GPT-4o",
                                                                "value": "gpt-4o",
                                                            },
                                                            {
                                                                "label": "o1-mini",
                                          
                                                                "value": "o1-mini",
                                                            },
                                                            {
                                                                "label": "o1-preview",
                                                                "value": "o1-preview",
                                                            },
                                                            {
                                                                "label": "GPT-3.5-turbo",
                                                                "value": "gpt-3.5-turbo",
                                                            },
                                                            {
                                                                "label": "Groq Mixtral",
                                                                "value": "mixtral-8x7b-32768",
                                                            },
                                                            {
                                                                "label": "Groq Llama 3.3 70b",
                                                                "value": "llama-3.3-70b-specdec",
                                                            },
                                                            {
                                                                "label": "Groq Gemma 7B",
                                                                "value": "gemma-7b-it",
                                                            },
                                                            {
                                                                "label": "Groq Deepseek 70B",
                                                                "value": "deepseek-r1-distill-llama-70b",
                                                            },
                                                        ],
                                                        value="gpt-4o-mini",
                                                        className="mb-2",
                                                    ),
                                                    id="model-selection-collapse",
                                                    is_open=False,
                                                ),
                                            ],
                                            xs=12,
                                            md=4,
                                        ),
                                        # File Upload Column
                                        dbc.Col(
                                            [
                                                html.H5(
                                                    "Dataset Upload", className="mb-2"
                                                ),
                                                dcc.Upload(
                                                    id="upload-data",
                                                    children=html.Div(
                                                        [
                                                            "Drag and Drop or ",
                                                            html.A(
                                                                "Select a CSV/Excel File"
                                                            ),
                                                        ]
                                                    ),
                                                    style={
                                                        "width": "100%",
                                                        "height": "120px",  # Increased height
                                                        "lineHeight": "120px",  # Adjusted line height
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "backgroundColor": COLORS[
                                                            "background"
                                                        ],
                                                    },
                                                    multiple=False,
                                                ),
                                                html.Div(
                                                    id="upload-status", className="mt-2"
                                                ),
                                            ],
                                            xs=12,
                                            md=5,
                                        ),
                                        # Analysis Controls Column
                                        dbc.Col(
                                            [
                                                html.H5("\u00a0", className="mb-2"),
                                                dbc.Button(
                                                    "Analyze Data",
                                                    id="analyze-button",
                                                    color="primary",
                                                    className="w-100 mt-2",
                                                    disabled=True,
                                                ),
                                                dcc.Dropdown(
                                                    id="kpi-selector",
                                                    multi=True,
                                                    placeholder="Select KPIs of interest...",
                                                    className="mt-2",
                                                    style={"width": "100%"},
                                                ),
                                                dbc.Checkbox(
                                                    id="viz-only-checkbox",
                                                    label="Add text insights (slower)",
                                                    value=False,
                                                    className="mt-2",
                                                    style={"color": "#6c757d"},
                                                ),
                                            ],
                                            xs=12,
                                            md=3,
                                            className="d-flex align-items-end flex-column",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Results Section
                dbc.Row(
                    [
                        # Filters Sidebar
                        dbc.Col(
                            [
                                dbc.Collapse(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Inferred Data Filters"),
                                            dbc.CardBody(id="filter-controls"),
                                        ],
                                        className="sticky-top",
                                    ),
                                    id="sidebar-collapse",
                                    is_open=False,
                                ),
                            ],
                            width=2,
                            style={"paddingRight": "20px"},
                        ),
                        # Main Content
                        dbc.Col(
                            [
                                dbc.Spinner(
                                    html.Div(
                                        id="results-container",
                                        style={"minHeight": "200px"},
                                    ),
                                    color="primary",
                                    type="border",
                                    fullscreen=False,
                                )
                            ],
                            width=10,
                        ),
                    ],
                    className="g-0",
                ),
                # Spacer div to push footer down
                html.Div(style={"flex": "1"}),
                # Footer
                html.Footer(
                    dbc.Row(
                        [
                            # Left column (empty now)
                            dbc.Col([], width=4),
                            # Center column with text and link
                            dbc.Col(
                                [
                                    html.P(
                                        [
                                            "AI Dashboard Builder is open source",
                                            html.A(
                                                children=[
                                                    html.I(
                                                        className="fa fa-github",
                                                        **{"aria-hidden": "true"},
                                                    ),
                                                    " Fork it or contribute on the project repo",
                                                ],
                                                href="https://github.com/pnmartinez/ai-dashboard-builder",
                                                target="_blank",
                                                style={
                                                    "color": COLORS["primary"],
                                                    "textDecoration": "none",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                        className="text-center mb-0",
                                        style={
                                            "color": COLORS["text_secondary"],
                                            "fontSize": "0.9rem",
                                        },
                                    )
                                ],
                                width=4,
                            ),
                            # Right column (empty)
                            dbc.Col([], width=4),
                        ],
                        className="py-2",
                        style={
                            "borderTop": f'1px solid {COLORS["divider"]}',
                            "backgroundColor": COLORS["background"],
                            "width": "100%",
                        },
                    )
                ),
            ],
            style={
                "minHeight": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "backgroundColor": COLORS["background"],
                "position": "relative",
            },
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Expanded View")),
                dbc.ModalBody(
                    dcc.Graph(id="modal-figure", config={"displayModeBar": True}),
                    style={"height": "80vh"},
                ),
            ],
            id="figure-modal",
            size="xl",
            is_open=False,
        ),
    ],
    style={"backgroundColor": COLORS["background"], "minHeight": "100vh"},
)


# --- 7. CALLBACKS ---
# Provider and API Key Management
@app.callback(
    [
        Output("api-key-collapse", "is_open"),
        Output("model-selection-collapse", "is_open"),
        Output("api-key-input", "value"),
        Output("api-key-input", "placeholder"),
    ],
    [Input("llm-provider", "value"), Input("model-selection", "value")],
    [State("api-key-input", "value")]  # Add State to preserve user input
)
def toggle_api_key(provider: str, model: str, current_key: str) -> Tuple[bool, bool, str, str]:
    """Toggle visibility and populate API key input based on provider selection."""
    if provider != "external":
        return False, False, "", "Enter API Key"

    # Only get key from env if there's no user-entered key
    if not current_key:
        api_key = get_api_key_from_env_file(model)
        if api_key:
            return True, True, api_key, "API KEY loaded"
        return True, True, "", "Enter API Key"
    
    # Preserve user-entered key
    return True, True, current_key, "API KEY entered"

@app.callback(
    Output("api-key-input", "value", allow_duplicate=True),
    Input("api-key-input", "value"),
    prevent_initial_call=True
)
def update_api_key(value: str) -> str:
    """Handle user input in API key field."""
    return value


# File Upload and Preview
@app.callback(
    [
        Output("data-store", "data"),
        Output("upload-status", "children"),
        Output("analyze-button", "disabled"),
        Output("upload-data", "style"),
    ],
    Input("upload-data", "contents"),
    [State("upload-data", "filename"), State("upload-data", "style")],
    prevent_initial_call=True,
)
def handle_upload(contents: str, filename: str, current_style: Dict) -> Tuple:
    """Process uploaded data file and prepare it for analysis."""
    if contents is None:
        return no_update, no_update, True, current_style

    try:
        # File processing logic here
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Check file extension and read file
        file_extension = filename.lower().split(".")[-1]
        if file_extension not in ["csv", "xlsx", "xls"]:
            return (
                None,
                html.Div(
                    "Please upload a CSV or Excel file",
                    style={"color": COLORS["error"]},
                ),
                True,
                current_style,
            )

        if file_extension == "csv":
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            df = pd.read_excel(BytesIO(decoded))

        if df.empty:
            return (
                None,
                html.Div(
                    "The uploaded file is empty", style={"color": COLORS["error"]}
                ),
                True,
                current_style,
            )

        # Apply smart numeric conversion
        df = smart_numeric_conversion(df)

        # Create preview controls
        preview_controls = dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Limit Rows:", className="me-2"),
                        dbc.Input(
                            id="preview-rows-input",
                            type="number",
                            min=1,
                            max=len(df),
                            value=len(df),
                            style={"width": "100px"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Label("Limit Columns:", className="me-2"),
                        dbc.Input(
                            id="preview-cols-input",
                            type="number",
                            min=1,
                            max=len(df.columns),
                            value=len(df.columns),
                            style={"width": "100px"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "Update Limits",
                            id="update-preview-button",
                            color="secondary",
                            size="sm",
                            className="ms-2",
                        )
                    ],
                    width="auto",
                ),
            ],
            className="mb-3 align-items-center",
        )

        # Create initial preview table
        preview_df = df.head(10)
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
            style={"backgroundColor": "white"},
        )

        # Store data
        data_store = {
            "full_data": df.to_json(date_format="iso", orient="split"),
            "row_limit": 10,
            "col_limit": len(df.columns),
            "filename": filename,
        }

        # Hide upload component
        hidden_style = {**current_style, "display": "none"}

        # Add import viz specs button
        import_button = html.Div(
            [
                dbc.Button(
                    [
                        html.I(className="fas fa-file-import me-2"),
                        "Import Previous Viz Specs",
                    ],
                    id="import-viz-specs-button",
                    color="link",
                    className="p-0",
                    style={
                        "color": "#6c757d",
                        "fontSize": "0.8rem",
                        "textDecoration": "none",
                        "opacity": "0.7",
                    },
                ),
                dbc.Tooltip(
                    "Advanced option: Reuse previously generated visualization specifications",
                    target="import-viz-specs-button",
                    placement="right",
                ),
                html.Div(id="viz-specs-list", style={"display": "none"}),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Select Visualization Specifications"),
                        dbc.ModalBody(id="viz-specs-modal-content"),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close", id="close-viz-specs-modal", className="ms-auto"
                            )
                        ),
                    ],
                    id="viz-specs-modal",
                    size="lg",
                ),
            ],
            className="mt-2",
        )

        return (
            json.dumps(data_store),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                f"Loaded: {filename}", style={"color": COLORS["info"]}
                            ),
                            html.Button(
                                "Change File",
                                id="change-file-button",
                                className="mt-2 mb-3 btn btn-outline-secondary btn-sm",
                                n_clicks=0,
                            ),
                            import_button,
                        ],
                        id="file-info-container",
                    ),
                    html.Div(
                        [
                            html.H6("Data Preview:", className="mt-3"),
                            preview_controls,
                            html.Div(
                                id="preview-table-container",
                                children=[
                                    preview_table,
                                    html.Div(
                                        f"Using {len(preview_df)} of {len(df)} rows and {len(preview_df.columns)} of {len(df.columns)} columns",
                                        className="mt-2",
                                        style={"color": COLORS["text_secondary"]},
                                    ),
                                ],
                                style={
                                    "overflowX": "auto",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                },
                            ),
                            html.Div(
                                f"Total Dataset: {len(df)} rows, {len(df.columns)} columns",
                                className="mt-2",
                                style={"color": COLORS["text_secondary"]},
                            ),
                        ],
                        id="preview-section",
                    ),
                ]
            ),
            False,
            hidden_style,
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return (
            None,
            html.Div(f"Error: {str(e)}", style={"color": COLORS["error"]}),
            True,
            current_style,
        )


# Preview Table Updates
@app.callback(
    [
        Output("preview-table-container", "children"),
        Output("data-store", "data", allow_duplicate=True),
    ],
    [Input("update-preview-button", "n_clicks")],
    [
        State("preview-rows-input", "value"),
        State("preview-cols-input", "value"),
        State("data-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_preview(
    n_clicks: int, rows: int, cols: int, json_data: str
) -> Tuple[List, str]:
    """Update the data preview based on user-specified row and column limits."""
    if not n_clicks or not json_data:
        raise PreventUpdate

    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")

        rows = max(1, min(rows, len(df))) if rows else 10
        cols = max(1, min(cols, len(df.columns))) if cols else 10

        data_store["row_limit"] = rows
        data_store["col_limit"] = cols

        preview_df = df.head(rows).iloc[:, :cols]
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
            style={"backgroundColor": "white"},
        )

        return [
            [
                preview_table,
                html.Div(
                    f"Using {len(preview_df)} of {len(df)} rows and {len(preview_df.columns)} of {len(df.columns)} columns",
                    className="mt-2",
                    style={"color": COLORS["text_secondary"]},
                ),
            ],
            json.dumps(data_store),
        ]

    except Exception as e:
        logger.error(f"Preview update error: {str(e)}")
        return html.Div(
            f"Error updating preview: {str(e)}", style={"color": COLORS["error"]}
        ), no_update


# File Change Management
@app.callback(
    [
        Output("upload-data", "style", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Output("upload-status", "children", allow_duplicate=True),
        Output("analyze-button", "disabled", allow_duplicate=True),
        Output("viz-state", "data", allow_duplicate=True),
    ],
    Input("change-file-button", "n_clicks"),
    State("upload-data", "style"),
    prevent_initial_call=True,
)
def change_file(n_clicks: int, current_style: Dict) -> Tuple:
    """Handle file change request."""
    if n_clicks:
        visible_style = {**current_style, "display": "block"}
        return visible_style, None, "", True, None
    return no_update, no_update, no_update, no_update, no_update


# Visualization Specs Import
@app.callback(
    [
        Output("viz-specs-modal", "is_open"),
        Output("viz-specs-modal-content", "children"),
    ],
    [
        Input("import-viz-specs-button", "n_clicks"),
        Input("close-viz-specs-modal", "n_clicks"),
    ],
    [State("viz-specs-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_viz_specs_modal(
    import_clicks: Optional[int], close_clicks: Optional[int], is_open: bool
) -> Tuple[bool, Optional[html.Div]]:
    """Toggle and populate the visualization specifications import modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "import-viz-specs-button":
        # Update the viz_specs_dir path to be relative to src folder
        viz_specs_dir = os.path.join(
            BASE_DIR, "llm_responses"
        )  # BASE_DIR is already src directory
        viz_specs_files = glob.glob(os.path.join(viz_specs_dir, "viz_specs_*.json"))

        if not viz_specs_files:
            return True, html.Div(
                "No visualization specifications found", className="text-muted"
            )

        file_list = []
        for file_path in viz_specs_files:
            try:
                with open(file_path, "r") as f:
                    specs = json.load(f)
                    timestamp_str = specs.get("timestamp", "Unknown")
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        formatted_timestamp = timestamp_str
                        timestamp = datetime.min

                    # Get benchmark scores if available
                    benchmark_scores = specs.get("benchmark_scores", {})
                    overall_score = benchmark_scores.get("overall_score", None)
                    validity = benchmark_scores.get("validity", None)
                    relevance = benchmark_scores.get("relevance", None)
                    usefulness = benchmark_scores.get("usefulness", None)
                    diversity = benchmark_scores.get("diversity", None)
                    redundancy = benchmark_scores.get("redundancy", None)

                    file_list.append(
                        {
                            "path": os.path.relpath(file_path, BASE_DIR),
                            "timestamp": timestamp,
                            "display_data": {
                                "timestamp": formatted_timestamp,
                                "model": specs.get("model", "Unknown"),
                                "provider": specs.get("provider", "Unknown"),
                                "dataset_filename": specs.get("dataset_filename", "Unknown dataset"),
                                "scores": {
                                    "overall": overall_score,
                                    "validity": validity,
                                    "relevance": relevance,
                                    "usefulness": usefulness,
                                    "diversity": diversity,
                                    "redundancy": redundancy
                                }
                            },
                        }
                    )
            except Exception as e:
                logger.error(f"Error reading viz specs file {file_path}: {str(e)}")
                continue

        file_list.sort(key=lambda x: x["timestamp"], reverse=True)

        list_items = [
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H6(
                                f"Generated: {item['display_data']['timestamp']}",
                                className="mb-1",
                            ),
                            html.Small(
                                [
                                    f"Model: {item['display_data']['model']} ({item['display_data']['provider']})",
                                    html.Br(),
                                    f"For dataset: {item['display_data']['dataset_filename']}",
                                    html.Br(),
                                    html.Div([
                                        html.Strong("Benchmark Scores:", className="mt-2"),
                                        html.Div([
                                            html.Span(
                                                f"Overall: {item['display_data']['scores']['overall']:.2f}" if item['display_data']['scores']['overall'] is not None else "Overall: N/A",
                                                style={"marginRight": "10px", "color": "#28a745"}
                                            ),
                                            html.Span(
                                                f"Validity: {item['display_data']['scores']['validity']:.2f}" if item['display_data']['scores']['validity'] is not None else "Validity: N/A",
                                                style={"marginRight": "10px"}
                                            ),
                                            html.Span(
                                                f"Relevance: {item['display_data']['scores']['relevance']:.2f}" if item['display_data']['scores']['relevance'] is not None else "Relevance: N/A",
                                                style={"marginRight": "10px"}
                                            ),
                                            html.Br(),
                                            html.Span(
                                                f"Usefulness: {item['display_data']['scores']['usefulness']:.2f}" if item['display_data']['scores']['usefulness'] is not None else "Usefulness: N/A",
                                                style={"marginRight": "10px"}
                                            ),
                                            html.Span(
                                                f"Diversity: {item['display_data']['scores']['diversity']:.2f}" if item['display_data']['scores']['diversity'] is not None else "Diversity: N/A",
                                                style={"marginRight": "10px"}
                                            ),
                                            html.Span(
                                                f"Redundancy: {item['display_data']['scores']['redundancy']:.2f}" if item['display_data']['scores']['redundancy'] is not None else "Redundancy: N/A",
                                            ),
                                        ], style={"fontSize": "0.85em", "color": "#666"})
                                    ]) if any(score is not None for score in item['display_data']['scores'].values()) else None
                                ],
                                className="text-muted",
                            ),
                        ]
                    ),
                    dbc.Button(
                        "Use",
                        id={"type": "use-viz-specs", "index": item["path"]},
                        color="primary",
                        size="sm",
                        className="ms-auto",
                    ),
                ],
                className="d-flex justify-content-between align-items-center",
            )
            for item in file_list
        ]

        return True, dbc.ListGroup(list_items)

    return False, None


# Visualization Specs Usage
@app.callback(
    [
        Output("data-store", "data", allow_duplicate=True),
        Output("analyze-button", "n_clicks", allow_duplicate=True),
        Output("viz-specs-modal", "is_open", allow_duplicate=True),
        Output("viz-state", "data"),
    ],
    Input({"type": "use-viz-specs", "index": ALL}, "n_clicks"),
    [State("data-store", "data"), State("analyze-button", "n_clicks")],
    prevent_initial_call=True,
)
def use_viz_specs(
    n_clicks: List[Optional[int]], current_data: str, current_clicks: Optional[int]
) -> Tuple:
    """Import and apply previously saved visualization specifications."""
    ctx = dash.callback_context
    if not any(n_clicks):
        raise PreventUpdate

    try:
        triggered = ctx.triggered[0]
        file_path = eval(triggered["prop_id"].split(".")[0] + '"}')["index"]
        if not file_path.endswith(".json"):
            file_path = f"{file_path}.json"

        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Visualization specs file not found: {file_path}")

        with open(file_path, "r") as f:
            viz_specs = json.load(f)

        current_data = json.loads(current_data)
        current_data["imported_viz_specs"] = viz_specs["visualization_specs"]

        return json.dumps(current_data), (current_clicks or 0) + 1, False, True

    except Exception as e:
        logger.error(f"Error in use_viz_specs: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise PreventUpdate


# Data Analysis Pipeline
@app.long_callback(
    [
        Output("results-container", "children"),
        Output("dashboard-rendered", "data"),
        Output("data-store", "data", allow_duplicate=True),
    ],
    Input("analyze-button", "n_clicks"),
    [
        State("data-store", "data"),
        State("llm-provider", "value"),
        State("api-key-input", "value"),
        State("model-selection", "value"),
        State("viz-only-checkbox", "value"),
        State("kpi-selector", "value"),
    ],
    prevent_initial_call=True,
    running=[
        (Output("analyze-button", "disabled"), True, False),
        (Output("upload-data", "disabled"), True, False),
        (Output("llm-provider", "disabled"), True, False),
        (Output("api-key-input", "disabled"), True, False),
        (Output("model-selection", "disabled"), True, False),
        (Output("kpi-selector", "disabled"), True, False),
    ],
    progress=[Output("upload-status", "children")],
)
def analyze_data(
    set_progress,
    n_clicks: int,
    json_data: str,
    provider: str,
    input_api_key: str,
    model: str,
    include_text: bool,
    kpis: List[str],
) -> Tuple[html.Div, bool, str]:
    """Process the uploaded dataset and generate visualizations and analysis."""
    if not n_clicks or not json_data:
        raise PreventUpdate

    try:
        # Prioritize UI-provided API key over environment variable
        api_key = input_api_key or get_api_key_from_env_file(model)
        
        if provider == "external" and not api_key:
            return html.Div([
                dbc.Alert(
                    "API key is required for external provider",
                    color="danger",
                    dismissable=True,
                    style={"maxWidth": "800px", "margin": "20px auto"}
                )
            ]), False, json_data

        data_store = json.loads(json_data)
        df_full = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")
        filename = data_store.get("filename", "unknown_file")
        df = df_full

        set_progress(
            html.Div(
                "Initializing analysis pipeline...", 
                style={"color": COLORS["info"]}
            )
        )

        try:
            if provider == "local":
                pipeline = LLMPipeline(model_name="llama3.1", use_local=True)
            else:
                os.environ["LLM_API_KEY"] = api_key
                pipeline = LLMPipeline(model_name=model, use_local=False)

            if include_text:
                set_progress(
                    html.Div(
                        "1/5 Analyzing dataset... (Rate limiting in effect)",
                        style={"color": COLORS["info"]}
                    )
                )
                analysis = pipeline.analyze_dataset(df, kpis)
                if isinstance(analysis, str) and analysis.startswith("Error:"):
                    return html.Div([
                        dbc.Alert(
                            analysis[7:],  # Remove "Error: " prefix
                            color="danger",
                            dismissable=True,
                            style={"maxWidth": "800px", "margin": "20px auto"}
                        ),
                        dbc.Alert(
                            [
                                html.I(className="fas fa-info-circle me-2"),
                                "Try switching to a different model or API key.",
                            ],
                            color="info",
                            dismissable=True,
                            style={"maxWidth": "800px", "margin": "20px auto"}
                        )
                    ]), False, json_data
            else:
                analysis = None

            set_progress(
                html.Div(
                    "2/5 Generating visualization suggestions... (Rate limiting in effect)",
                    style={"color": COLORS["info"]}
                )
            )
            
            viz_specs = pipeline.suggest_visualizations(df, kpis, filename=filename)
            
            # Check if viz_specs is an error message
            if isinstance(viz_specs, str) and viz_specs.startswith("Error:"):
                return html.Div([
                    dbc.Alert(
                        viz_specs[7:],  # Remove "Error: " prefix
                        color="danger",
                        dismissable=True,
                        style={"maxWidth": "800px", "margin": "20px auto"}
                    ),
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "Try switching to a different model or API key.",
                        ],
                        color="info",
                        dismissable=True,
                        style={"maxWidth": "800px", "margin": "20px auto"}
                    )
                ]), False, json_data

            set_progress(
                html.Div(
                    "3/5 Creating visualizations...",
                    style={"color": COLORS["info"]}
                )
            )

            # Create dashboard builder and generate figures
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)

            if include_text and analysis:
                set_progress(
                    html.Div(
                        "4/5 Generating insights summary...",
                        style={"color": COLORS["info"]}
                    )
                )
                summary = pipeline.summarize_analysis(analysis, viz_specs)
            else:
                summary = None

            # Create visualization components
            components = [
                dbc.Card(
                    [
                        dbc.CardHeader(html.H3("Dashboard", className="mb-0")),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            [
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            html.H5(
                                                                                viz_specs[viz_id]["title"],
                                                                                className="mb-0",
                                                                            ),
                                                                            className="pe-4",
                                                                        ),
                                                                        dbc.Col(
                                                                            dbc.Button(
                                                                                html.I(className="fa fa-expand"),
                                                                                id={
                                                                                    "type": "maximize-btn",
                                                                                    "index": i,
                                                                                },
                                                                                color="link",
                                                                                size="sm",
                                                                                style={
                                                                                    "color": COLORS["text_secondary"],
                                                                                    "border": "none",
                                                                                    "backgroundColor": "transparent",
                                                                                    "padding": "4px 8px",
                                                                                },
                                                                            ),
                                                                            width="auto",
                                                                            className="ps-2",
                                                                        ),
                                                                    ],
                                                                    className="align-items-center g-0",
                                                                )
                                                            ]
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Div(
                                                                            [
                                                                                dcc.Graph(
                                                                                    id={
                                                                                        "type": "viz",
                                                                                        "index": i,
                                                                                    },
                                                                                    figure=fig,
                                                                                    config={
                                                                                        "displayModeBar": False
                                                                                    },
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Small(
                                                                                            viz_specs[viz_id].get('relationship_text', ''),
                                                                                            style={
                                                                                                "color": COLORS["primary"] if viz_specs[viz_id].get('relationship_significance', False) else COLORS["text_secondary"],
                                                                                                "fontStyle": "italic",
                                                                                                "display": "block",
                                                                                                "marginTop": "8px",
                                                                                                "padding": "8px",
                                                                                                "backgroundColor": f"{COLORS['background']}",
                                                                                                "borderRadius": "4px",
                                                                                                "border": f"1px solid {COLORS['divider']}"
                                                                                            } if viz_specs[viz_id].get('relationship_text') else {"display": "none"}
                                                                                        )
                                                                                    ]
                                                                                ),
                                                                            ],
                                                                            id={
                                                                                "type": "chart-content",
                                                                                "index": i,
                                                                            },
                                                                            className="chart-container",
                                                                        ),
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                )
                                            ],
                                            xs=12,
                                            md=6,
                                        )
                                        for i, (viz_id, (fig, code)) in enumerate(figures.items())
                                    ]
                                )
                            ]
                        ),
                    ],
                    className="mb-4",
                )
            ]

            if include_text and analysis and summary:
                components.extend([
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H3("Key Insights", className="mb-0")),
                            dbc.CardBody(
                                dcc.Markdown(
                                    summary,
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "padding": "1rem",
                                        "borderRadius": "5px",
                                    },
                                )
                            ),
                        ],
                        className="mb-4",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H3("Dataset Analysis", className="mb-0")),
                            dbc.CardBody(
                                dcc.Markdown(
                                    analysis,
                                    style={
                                        "backgroundColor": COLORS["background"],
                                        "padding": "1rem",
                                        "borderRadius": "5px",
                                    },
                                )
                            ),
                        ]
                    ),
                ])

            # Always return a tuple with all three elements
            return html.Div(components), True, json_data

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                error_msg = "⚠️ API Quota Exceeded: Your API key has exceeded its quota. Please check your billing details or try a different API key."
            elif "rate limit" in error_msg.lower():
                error_msg = "⚠️ Rate Limit: Too many requests. Please try again in a few moments."
            
            return html.Div([
                dbc.Alert(
                    error_msg,
                    color="danger",
                    dismissable=True,
                    style={"maxWidth": "800px", "margin": "20px auto"}
                ),
                dbc.Alert(
                    [
                        html.I(className="fas fa-info-circle me-2"),
                        "Try switching to a different model or API key.",
                    ],
                    color="info",
                    dismissable=True,
                    style={"maxWidth": "800px", "margin": "20px auto"}
                )
            ]), False, json_data

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return html.Div([
            dbc.Alert(
                f"Error during analysis: {str(e)}",
                color="danger",
                dismissable=True,
                style={"maxWidth": "800px", "margin": "20px auto"}
            )
        ]), False, json_data


# Tab Switching
@app.callback(
    [
        Output({"type": "chart-content", "index": MATCH}, "style"),
        Output({"type": "code-content", "index": MATCH}, "style"),
    ],
    Input({"type": "tabs", "index": MATCH}, "active_tab"),
    prevent_initial_call=True,
)
def switch_tab(active_tab: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Switch between chart and code views in visualization tabs."""
    if active_tab and "chart-tab" in active_tab:
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


# Code Copying (Client-side callback)
app.clientside_callback(
    """
    function(n_clicks, code_content) {
        if (n_clicks) {
            const code = code_content.props.children;
            navigator.clipboard.writeText(code);
            return "Copied!";
        }
        return "Copy Code";
    }
    """,
    Output({"type": "copy-btn", "index": MATCH}, "children"),
    Input({"type": "copy-btn", "index": MATCH}, "n_clicks"),
    State({"type": "code-content", "index": MATCH}, "children"),
    prevent_initial_call=True,
)


# Preview Visibility
@app.callback(
    Output("preview-section", "style"),
    [Input("viz-state", "data"), Input("change-file-button", "n_clicks")],
    prevent_initial_call=True,
)
def toggle_preview_visibility(viz_active: bool, change_clicks: int) -> Dict[str, str]:
    """Toggle visibility of the data preview section."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "viz-state" and viz_active:
        return {"display": "none"}
    elif trigger_id == "change-file-button":
        return {"display": "block"}

    return dash.no_update


# Button Text Update
@app.callback(Output("analyze-button", "children"), Input("dashboard-rendered", "data"))
def update_button_text(dashboard_rendered: bool) -> str:
    """Update the analyze button text based on dashboard state."""
    return "Regenerate Dashboard" if dashboard_rendered else "Analyze Data"


# Filter Controls Creation
@app.callback(
    [Output("filter-controls", "children"), Output("sidebar-collapse", "is_open")],
    [Input("dashboard-rendered", "data")],
    [State("data-store", "data"), State("filter-state", "data")],
    prevent_initial_call=True,
)
def create_filter_controls(
    dashboard_rendered: bool, json_data: str, filter_state: Optional[Dict]
) -> Tuple[List, bool]:
    """Create filter controls based on the dataset columns."""
    if not dashboard_rendered or not json_data:
        return [], False

    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")

        filters = []
        temporal_col = None

        # Find temporal column with improved detection
        def is_temporal_column(series: pd.Series) -> bool:
            """Check if a series contains valid datetime values."""
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return True
                if pd.api.types.is_numeric_dtype(series):
                    return False
                if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                    sample = series.dropna().head(10)
                    if len(sample) == 0:
                        return False
                    success_count = 0
                    for val in sample:
                        if not isinstance(val, str):
                            continue
                        try:
                            pd.to_datetime(val)
                            success_count += 1
                        except (ValueError, TypeError):
                            continue
                    return success_count / len(sample) >= 0.8
                return False
            except Exception:
                return False

        # First check columns that are already datetime
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            temporal_col = datetime_cols[0]
        else:
            # Then check other columns that might contain datetime values
            for col in df.columns:
                if is_temporal_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if not df[col].isna().all():
                            temporal_col = col
                            break
                    except Exception:
                        continue

        # Add temporal filter if found
        if temporal_col:
            valid_dates = df[temporal_col].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()

                if pd.notna(min_date) and pd.notna(max_date):
                    min_date_str = min_date.strftime("%Y-%m-%d")
                    max_date_str = max_date.strftime("%Y-%m-%d")

                    # Get current filter values from filter state if available
                    current_start = min_date_str
                    current_end = max_date_str
                    if filter_state and "temporal" in filter_state:
                        current_start = filter_state["temporal"].get("start_date", min_date_str)
                        current_end = filter_state["temporal"].get("end_date", max_date_str)

                    filters.extend([
                        html.H6("Time Range", className="mt-3"),
                        dcc.DatePickerRange(
                            id="date-range-filter",
                            min_date_allowed=min_date_str,
                            max_date_allowed=max_date_str,
                            start_date=current_start,
                            end_date=current_end,
                            className="mb-3 w-100",
                        ),
                    ])

        # Add categorical filters
        categorical_cols = [
            col
            for col in df.columns
            if col != temporal_col and df[col].nunique() / len(df) < 0.05
        ]

        for col in categorical_cols:
            unique_values = sorted(df[col].dropna().unique())
            if len(unique_values) > 0:
                # Get current values from filter state
                current_values = []
                if filter_state and "categorical" in filter_state:
                    current_values = filter_state["categorical"].get(col, [])

                filters.extend([
                    html.H6(f"{col}", className="mt-3"),
                    dcc.Dropdown(
                        id={"type": "category-filter", "column": col},
                        options=[{"label": str(val), "value": str(val)} for val in unique_values],
                        value=current_values,
                        multi=True,
                        placeholder=f"Select {col}...",
                        className="mb-3",
                    ),
                ])

        if filters:
            filters.append(
                dbc.Button(
                    "Reset Filters",
                    id="reset-filters-button",
                    color="secondary",
                    size="sm",
                    className="mt-3 w-100",
                )
            )

            return html.Div(filters, style={"padding": "10px"}), True

        return [], False

    except Exception as e:
        logger.error(f"Error creating filters: {str(e)}")
        return [], False


# Filter State Management
@app.callback(
    Output("filter-state", "data"),
    [
        Input("reset-filters-button", "n_clicks"),
        Input({"type": "category-filter", "column": ALL}, "value"),
    ],
    [
        State({"type": "category-filter", "column": ALL}, "id"),
        State("filter-state", "data"),
    ],
    prevent_initial_call=True,
)
def update_filter_state(
    reset_clicks: int,
    category_values: List,
    category_ids: List,
    current_filter_state: Optional[Dict],
) -> Optional[Dict]:
    """Update the filter state based on user selections."""
    logger.info("=" * 50)
    logger.info("FILTER STATE UPDATE TRIGGERED")
    logger.info("=" * 50)
    
    ctx = dash.callback_context
    if not ctx.triggered:
        logger.info("No trigger for filter state update")
        raise PreventUpdate

    trigger = ctx.triggered[0]
    logger.info(f"Trigger: {trigger}")
    logger.info(f"Trigger prop_id: {trigger['prop_id']}")
    logger.info(f"Trigger value: {trigger['value']}")
    logger.info(f"Current filter state: {current_filter_state}")
    logger.info(f"Category values: {category_values}")
    logger.info(f"Category IDs: {category_ids}")

    # Handle reset button click
    if trigger["prop_id"] == "reset-filters-button.n_clicks":
        logger.info("Reset filters button clicked, clearing filter state")
        return None

    # Initialize filter state from current state or create new
    filter_state = current_filter_state.copy() if current_filter_state else {"temporal": {}, "categorical": {}}
    logger.info(f"Initial filter state: {filter_state}")

    # Handle categorical filter updates
    if category_values and category_ids:
        logger.info("Updating categorical filters")
        filter_state["categorical"] = {
            id["column"]: values
            for id, values in zip(category_ids, category_values)
            if values
        }
        logger.info(f"Set categorical filters: {filter_state['categorical']}")

    # Only return filter state if it contains actual filters
    if filter_state.get("temporal") or filter_state.get("categorical"):
        logger.info(f"Returning updated filter state: {filter_state}")
        return filter_state
    
    logger.info("No active filters, returning None")
    return None


# Date Range Filter Updates
@app.callback(
    Output("filter-state", "data", allow_duplicate=True),
    [
        Input("date-range-filter", "start_date"),
        Input("date-range-filter", "end_date"),
    ],
    State("filter-state", "data"),
    prevent_initial_call=True,
)
def update_date_range_filter(
    start_date: str,
    end_date: str,
    current_filter_state: Optional[Dict],
) -> Optional[Dict]:
    """Update filter state when date range changes."""
    logger.info("=" * 50)
    logger.info("DATE RANGE FILTER UPDATE TRIGGERED")
    logger.info("=" * 50)
    
    logger.info(f"Start date: {start_date}, End date: {end_date}")
    logger.info(f"Current filter state: {current_filter_state}")

    if not dash.callback_context.triggered:
        logger.info("No trigger for date range update")
        raise PreventUpdate

    # Initialize filter state from current state or create new
    filter_state = current_filter_state.copy() if current_filter_state else {"temporal": {}, "categorical": {}}
    
    # Update temporal filter
    if start_date is not None and end_date is not None:
        filter_state["temporal"] = {"start_date": start_date, "end_date": end_date}
        logger.info(f"Set temporal filter: {filter_state['temporal']}")
    else:
        filter_state["temporal"] = {}
        logger.info("Cleared temporal filter")
    
    # Return None if no filters are active
    if not filter_state["temporal"] and not filter_state.get("categorical"):
        logger.info("No active filters, returning None")
        return None
    
    logger.info(f"Returning updated filter state: {filter_state}")
    return filter_state


# Visualization Updates
@app.callback(
    Output({"type": "viz", "index": ALL}, "figure"),
    [Input("filter-state", "data")],
    [
        State("data-store", "data"),
        State({"type": "viz", "index": ALL}, "figure"),
    ],
    prevent_initial_call=True,
)
def update_visualizations(
    filter_state: Optional[Dict],
    json_data: str,
    current_figures: List,
) -> List:
    """Update all visualizations based on the current filter state."""
    logger.info("=" * 50)
    logger.info("VISUALIZATION UPDATE TRIGGERED")
    logger.info("=" * 50)
    
    logger.info(f"Received filter state: {filter_state}")
    logger.info(f"Current figures count: {len(current_figures) if current_figures else 0}")

    if not json_data:
        logger.warning("No data available for visualization update")
        raise PreventUpdate

    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")
        logger.info(f"Loaded dataframe with shape: {df.shape}")

        viz_specs = data_store.get("imported_viz_specs") or data_store.get(
            "visualization_specs"
        )
        if not viz_specs:
            logger.warning("No visualization specifications found in data store")
            return current_figures

        if not filter_state:
            logger.info("No filter state, creating visualizations with full dataset")
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            logger.info(f"Created {len(figures)} figures from full dataset")
            result = [list(figures.values())[i][0] for i in range(len(current_figures))]
            logger.info(f"Returning {len(result)} figures")
            return result

        logger.info("Applying filters to dataset")
        filtered_df = apply_filters(df, filter_state)
        logger.info(f"Dataset after filtering: {filtered_df.shape}")
        
        dashboard_builder = DashboardBuilder(filtered_df, COLORS)
        figures = dashboard_builder.create_all_figures(viz_specs)
        logger.info(f"Created {len(figures)} figures from filtered dataset")
        result = [list(figures.values())[i][0] for i in range(len(current_figures))]
        logger.info(f"Returning {len(result)} figures")
        return result

    except Exception as e:
        logger.error(f"Error updating visualizations: {str(e)}", exc_info=True)
        logger.error("Full traceback:", exc_info=True)
        return current_figures


# Date Range Updates
@app.callback(
    [
        Output("date-range-filter", "start_date"),
        Output("date-range-filter", "end_date"),
    ],
    Input("dashboard-rendered", "data"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def update_date_range(dashboard_rendered: bool, json_data: str) -> Tuple[str, str]:
    """Update date range filter with initial values."""
    if not dashboard_rendered or not json_data:
        raise PreventUpdate

    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")

        # Find temporal column and get date range
        for col in df.columns:
            try:
                temp_series = pd.to_datetime(df[col], errors="coerce")
                if not temp_series.isna().all():
                    df[col] = temp_series
                    min_date = df[col].min().strftime("%Y-%m-%d")
                    max_date = df[col].max().strftime("%Y-%m-%d")
                    return min_date, max_date
            except Exception:
                continue

        return dash.no_update, dash.no_update

    except Exception as e:
        logger.error(f"Error updating date range: {str(e)}")
        return dash.no_update, dash.no_update


# KPI Selector Population
@app.callback(
    Output("kpi-selector", "options"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_kpi_selector(json_data: str) -> List[Dict[str, str]]:
    """Update KPI selector options based on loaded dataset columns."""
    if not json_data:
        return []

    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store["full_data"]), orient="split")
        return [{"label": col, "value": col} for col in df.columns]

    except Exception as e:
        logger.error(f"Error updating KPI selector: {str(e)}")
        return []


# Add these new callbacks at the end of the file
@app.callback(
    [Output("figure-modal", "is_open"), Output("modal-figure", "figure")],
    [Input({"type": "maximize-btn", "index": ALL}, "n_clicks")],
    [State({"type": "viz", "index": ALL}, "figure"), State("figure-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n_clicks, figures, is_open):
    """Handle maximizing/minimizing of figures."""
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return False, dash.no_update

    # Get the index of the clicked button
    button_id = ctx.triggered[0]["prop_id"]
    clicked_idx = json.loads(button_id.split(".")[0])["index"]
    
    # Get the corresponding figure
    figure = figures[clicked_idx]
    
    # Create a new figure with adjusted layout for the modal
    modal_figure = go.Figure(figure)
    
    # Update layout for modal
    modal_figure.update_layout(
        height=700,  # Slightly less than 80vh to ensure it fits
        margin=dict(l=20, r=20, t=30, b=20),
        autosize=True,
        showlegend=True,
        legend=dict(
            bgcolor="white",
            bordercolor="#FFD7D7",
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return not is_open, modal_figure


# Define styles as dictionaries for better maintainability
chart_container_style = {
    "position": "relative"  # To contain the maximize button
}

maximize_btn_style = {
    "opacity": "0",
    "transition": "opacity 0.2s",
    "position": "absolute",
    "top": "5px",
    "right": "5px",
    "backgroundColor": "transparent",
    ":hover": {"backgroundColor": "#f8f9fa"},
}
