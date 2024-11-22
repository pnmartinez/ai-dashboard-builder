"""
AI Dashboard Builder - A Dash application for automated dashboard creation using LLMs.

This module serves as the main entry point for the AI Dashboard Builder application.
It handles the web interface, data processing, and visualization generation using
various LLM providers.
"""

# --- 1. IMPORTS ---
# Standard library imports
import os
import io
from io import BytesIO
import json
import logging
import glob
import base64
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Third-party imports
import dash
from dash import Dash, html, dcc, Input, Output, State, no_update, long_callback, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import markdown2
from dotenv import load_dotenv
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

# Local imports
from llm_pipeline import LLMPipeline
from dashboard_builder import DashboardBuilder

# --- 2. CONSTANTS ---
# Color Palette - Based on matplotlib style with additional shades
COLORS = {
    'background': '#FFF9F0',      # Lightest - Main page background
    'surface': '#FFF4E2',         # Medium - Card backgrounds
    'plot_container': '#FFF2DE',  # Darker - Plot container background (adjusted)
    'plot_surface': '#FFF4E2',    # Plot surface background
    'grid': '#FFE4C4',           # Darker grid color (adjusted)
    'primary': '#ff5200',         # Vibrant orange
    'primary_light': '#E9540D',   # Lighter orange
    'primary_dark': '#b60505',    # Dark red
    'secondary': '#0095f1',       # Bright blue
    'accent': '#15bf14',          # Green
    'warning': '#ffc107',         # Amber
    'error': '#ff2525',           # Red
    'success': '#4caf50',         # Green
    'info': '#3A507C',           # Navy blue
    'text_primary': '#00234c',    # Dark navy text
    'text_secondary': '#2f4968',  # Lighter navy text
    'divider': '#FFF9F0',        # Light beige divider
    'shadow': 'rgba(0, 35, 76, 0.1)' # Based on text_primary
}

# Data preview limits
MAX_PREVIEW_ROWS = 1000
MAX_PREVIEW_COLS = 100

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 3. CONFIGURATION ---
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Dashboard')
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Setup diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# --- 4. APP INITIALIZATION ---
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=False,
    update_title=None,
    long_callback_manager=long_callback_manager,
    title="AI Dashboard Builder"
)

# --- 5. UTILITY FUNCTIONS ---
def get_api_key(model_name: str) -> str:
    """Get the appropriate API key based on the model name."""
    key_mapping = {
        'gpt': 'OPENAI_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'mixtral': 'GROQ_API_KEY',
        'groq': 'GROQ_API_KEY',
        'llama': 'GROQ_API_KEY',
        'gemma': 'GROQ_API_KEY'
    }
    
    for model_type, env_key in key_mapping.items():
        if model_type in model_name.lower():
            return os.getenv(env_key, '')
    return ''

def smart_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently convert string columns to numeric types where possible."""
    def clean_numeric_string(s: Any) -> Any:
        if pd.isna(s):
            return s
        if not isinstance(s, str):
            return s
            
        s = str(s).strip()
        s = re.sub(r'[($€£¥,)]', '', s)
        
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
            
        if s.endswith('%'):
            try:
                return float(s.rstrip('%')) / 100
            except:
                return s
                
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        if s and s[-1].upper() in multipliers:
            try:
                return float(s[:-1]) * multipliers[s[-1].upper()]
            except:
                return s
                
        return s

    def try_numeric_conversion(series: pd.Series) -> pd.Series:
        cleaned = series.map(clean_numeric_string)
        try:
            numeric = pd.to_numeric(cleaned, errors='coerce')
            na_ratio = numeric.isna().sum() / len(numeric)
            if na_ratio < 0.3:
                return numeric
        except:
            pass
        return series

    df_converted = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            continue
            
        try:
            pd.to_datetime(df[col], errors='raise')
            continue
        except:
            pass
            
        df_converted[col] = try_numeric_conversion(df[col])

    return df_converted

def apply_filters(df: pd.DataFrame, filter_state: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    if not filter_state:
        return df
        
    filtered_df = df.copy()
    
    temporal_filter = filter_state.get('temporal', {})
    if temporal_filter.get('start_date') and temporal_filter.get('end_date'):
        for col in df.columns:
            try:
                filtered_df[col] = pd.to_datetime(filtered_df[col])
                filtered_df = filtered_df[
                    (filtered_df[col] >= temporal_filter['start_date']) &
                    (filtered_df[col] <= temporal_filter['end_date'])
                ]
                break
            except:
                continue
    
    categorical_filters = filter_state.get('categorical', {})
    for col, values in categorical_filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
    
    return filtered_df

# --- 6. LAYOUT ---
app.layout = html.Div([
    # Data stores
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='viz-state', storage_type='memory'),
    dcc.Store(id='dashboard-rendered', storage_type='memory'),
    dcc.Store(id='filter-state', storage_type='memory'),
    
    # Main container
    dbc.Container(fluid=True, children=[
        # Header with elevation
        dbc.Card([
            html.A(
                html.H1('AI Dashboard Builder',
                    style={
                        'textAlign': 'center',
                        'color': COLORS['primary'],
                        'margin': '1rem 0',
                        'fontWeight': '300',
                        'letterSpacing': '0.5px'
                    }
                ),
                href='/',
                style={'textDecoration': 'none'}
            )
        ], 
        className="mb-4",
        style={
            'backgroundColor': COLORS['surface'],
            'borderRadius': '8px',
            'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
            'border': 'none'
        }),
        
        # Controls Section
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # LLM Provider Column
                    dbc.Col([
                        html.H5("LLM Provider", 
                               className="mb-3",
                               style={'fontWeight': '500', 'color': COLORS['text_primary']}),
                        dbc.RadioItems(
                            id='llm-provider',
                            options=[
                                {'label': ['Local ', html.A('(Ollama)', href='https://ollama.com/download', 
                                                          target='_blank',
                                                          style={'color': COLORS['primary']})], 
                                 'value': 'local'},
                                {'label': 'External API', 'value': 'external'}
                            ],
                            value='local',
                            className="mb-3",
                            inline=True,
                            style={
                                'gap': '1rem',
                                'color': COLORS['text_secondary']
                            }
                        ),
                        dbc.Collapse(
                            dbc.Input(
                                id='api-key-input',
                                type='password',
                                placeholder='Enter API Key',
                                className="mb-3",
                                style={
                                    'borderRadius': '4px',
                                    'border': f'1px solid {COLORS["divider"]}',
                                    'transition': 'border-color 0.15s ease-in-out'
                                }
                            ),
                            id='api-key-collapse',
                            is_open=False
                        ),
                        dbc.Collapse(
                            dbc.Select(
                                id='model-selection',
                                options=[
                                    {'label': 'GPT-4o-mini', 'value': 'gpt-4o-mini'},
                                    {'label': 'GPT-3.5-turbo', 'value': 'gpt-3.5-turbo'},
                                    {'label': 'Groq Mixtral', 'value': 'mixtral-8x7b-32768'},
                                    {'label': 'Groq Llama 3.1 70b', 'value': 'llama-3.1-70b-versatile'},
                                    {'label': 'Groq Gemma 7B', 'value': 'gemma-7b-it'},
                                ],
                                value='gpt-4o-mini',
                                className="mb-2"
                            ),
                            id='model-selection-collapse',
                            is_open=False
                        ),
                    ], xs=12, md=4),
                    
                    # File Upload Column with modern styling
                    dbc.Col([
                        html.H5("Dataset Upload", 
                               className="mb-3",
                               style={'fontWeight': '500', 'color': COLORS['text_primary']}),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2",
                                      style={'color': COLORS['primary']}),
                                html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select a File',
                                          style={'color': COLORS['primary'],
                                                'textDecoration': 'none',
                                                'fontWeight': '500'})
                                ])
                            ], style={'textAlign': 'center'}),
                            style={
                                'width': '100%',
                                'height': '120px',
                                'lineHeight': '1.5',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '8px',
                                'borderColor': COLORS['primary_light'],
                                'backgroundColor': f'{COLORS["primary"]}08',
                                'color': COLORS['text_secondary'],
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s ease'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mt-3"),
                    ], xs=12, md=5),
                    
                    # Analysis Controls Column
                    dbc.Col([
                        html.H5("\u00A0", className="mb-3"),
                        dbc.Button(
                            'Analyze Data',
                            id='analyze-button',
                            color='primary',
                            className='w-100 mb-3',
                            style={
                                'borderRadius': '4px',
                                'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                                'fontWeight': '500',
                                'textTransform': 'uppercase',
                                'letterSpacing': '0.5px'
                            },
                            disabled=True
                        ),
                        dcc.Dropdown(
                            id='kpi-selector',
                            multi=True,
                            placeholder="Select KPIs of interest...",
                            className="mt-2",
                            style={'width': '100%'}
                        ),
                        dbc.Checkbox(
                            id='viz-only-checkbox',
                            label="Visualizations only (faster process)",
                            value=True,
                            className="mt-2",
                            style={'color': '#6c757d'}
                        ),
                    ], xs=12, md=3),
                ])
            ])
        ],
        style={
            'backgroundColor': COLORS['surface'],
            'borderRadius': '8px',
            'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
            'border': 'none'
        },
        className="mb-4"),
        
        # Results Section
        dbc.Row([
            # Filters Sidebar
            dbc.Col([
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H6("Inferred Data Filters",
                                   className="mb-0",
                                   style={'fontWeight': '500',
                                         'color': COLORS['text_primary']})),
                        dbc.CardBody(id='filter-controls')
                    ],
                    style={
                        'backgroundColor': COLORS['surface'],
                        'borderRadius': '8px',
                        'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                        'border': 'none',
                        'position': 'sticky',
                        'top': '1rem'
                    }),
                    id="sidebar-collapse",
                    is_open=False,
                ),
            ], width=2),
            
            # Main Content
            dbc.Col([
                dbc.Spinner(
                    html.Div(
                        id='results-container',
                        style={'minHeight': '200px'}
                    ),
                    color=COLORS['primary'],
                    spinner_style={'width': '3rem', 'height': '3rem'}
                )
            ], width=10)
        ], className="g-0")
    ])
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# --- 7. CALLBACKS ---
# Provider and API Key Management
@app.callback(
    [Output('api-key-collapse', 'is_open'),
     Output('model-selection-collapse', 'is_open'),
     Output('api-key-input', 'value'),
     Output('api-key-input', 'placeholder')],
    [Input('llm-provider', 'value'),
     Input('model-selection', 'value')]
)
def toggle_api_key(provider: str, model: str) -> tuple:
    """Toggle visibility and populate API key input based on provider selection."""
    if provider != 'external':
        return False, False, '', 'Enter API Key'
    
    api_key = get_api_key(model)
    if api_key:
        return True, True, api_key, 'API KEY loaded'
    return True, True, '', 'Enter API Key'

# File Upload and Preview
@app.callback(
    [Output('data-store', 'data'),
     Output('upload-status', 'children'),
     Output('analyze-button', 'disabled'),
     Output('upload-data', 'style')],
    Input('upload-data', 'contents'),
    [State('upload-data', 'filename'),
     State('upload-data', 'style')],
    prevent_initial_call=True
)
def handle_upload(contents: str, filename: str, current_style: Dict) -> Tuple:
    """Process uploaded data file and prepare it for analysis."""
    if contents is None:
        return no_update, no_update, True, current_style
    
    try:
        # File processing logic here
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Check file extension and read file
        file_extension = filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            return None, html.Div('Please upload a CSV or Excel file', style={'color': COLORS['error']}), True, current_style
            
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            df = pd.read_excel(BytesIO(decoded))
            
        if df.empty:
            return None, html.Div('The uploaded file is empty', style={'color': COLORS['error']}), True, current_style
        
        # Apply smart numeric conversion
        df = smart_numeric_conversion(df)
        
        # Create preview controls
        preview_controls = dbc.Row([
            dbc.Col([
                html.Label("Limit Rows:", className="me-2"),
                dbc.Input(
                    id='preview-rows-input',
                    type='number',
                    min=1,
                    max=len(df),
                    value=len(df),
                    style={'width': '100px'}
                ),
            ], width='auto'),
            dbc.Col([
                html.Label("Limit Columns:", className="me-2"),
                dbc.Input(
                    id='preview-cols-input',
                    type='number',
                    min=1,
                    max=len(df.columns),
                    value=len(df.columns),
                    style={'width': '100px'}
                ),
            ], width='auto'),
            dbc.Col([
                dbc.Button(
                    "Update Limits",
                    id='update-preview-button',
                    color="secondary",
                    size="sm",
                    className="ms-2"
                )
            ], width='auto'),
        ], className="mb-3 align-items-center")

        # Create initial preview table
        preview_df = df.head(10)
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            style={'backgroundColor': 'white'}
        )
            
        # Store data
        data_store = {
            'full_data': df.to_json(date_format='iso', orient='split'),
            'row_limit': 10,
            'col_limit': len(df.columns),
            'filename': filename
        }
        
        # Hide upload component
        hidden_style = {**current_style, 'display': 'none'}

        # Add import viz specs button
        import_button = html.Div([
            dbc.Button(
                [
                    html.I(className="fas fa-file-import me-2"),
                    "Import Previous Viz Specs"
                ],
                id='import-viz-specs-button',
                color="link",
                className="p-0",
                style={
                    'color': '#6c757d',
                    'fontSize': '0.8rem',
                    'textDecoration': 'none',
                    'opacity': '0.7'
                }
            ),
            dbc.Tooltip(
                "Developer option: Reuse previously generated visualization specifications",
                target='import-viz-specs-button',
                placement='right'
            ),
            html.Div(id='viz-specs-list', style={'display': 'none'}),
            dbc.Modal(
                [
                    dbc.ModalHeader("Select Visualization Specifications"),
                    dbc.ModalBody(id='viz-specs-modal-content'),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-viz-specs-modal", className="ms-auto")
                    ),
                ],
                id="viz-specs-modal",
                size="lg",
            )
        ], className="mt-2")
            
        return (
            json.dumps(data_store),
            html.Div([
                html.Div([
                    html.Div(f'Loaded: {filename}', style={'color': COLORS['info']}),
                    html.Button(
                        'Change File',
                        id='change-file-button',
                        className='mt-2 mb-3 btn btn-outline-secondary btn-sm',
                        n_clicks=0
                    ),
                    import_button
                ], id='file-info-container'),
                
                html.Div([
                    html.H6("Data Preview:", className="mt-3"),
                    preview_controls,
                    html.Div(
                        id='preview-table-container',
                        children=[
                            preview_table,
                            html.Div(
                                f"Using {len(preview_df)} of {len(df)} rows and {len(preview_df.columns)} of {len(df.columns)} columns",
                                className="mt-2",
                                style={'color': COLORS['text_secondary']}
                            )
                        ],
                        style={
                            'overflowX': 'auto',
                            'maxHeight': '300px',
                            'overflowY': 'auto'
                        }
                    ),
                    html.Div(
                        f"Total Dataset: {len(df)} rows, {len(df.columns)} columns",
                        className="mt-2",
                        style={'color': COLORS['text_secondary']}
                    )
                ], id='preview-section')
            ]),
            False,
            hidden_style
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return None, html.Div(f'Error: {str(e)}', style={'color': COLORS['error']}), True, current_style

# Preview Table Updates
@app.callback(
    [Output('preview-table-container', 'children'),
     Output('data-store', 'data', allow_duplicate=True)],
    [Input('update-preview-button', 'n_clicks')],
    [State('preview-rows-input', 'value'),
     State('preview-cols-input', 'value'),
     State('data-store', 'data')],
    prevent_initial_call=True
)
def update_preview(n_clicks: int, rows: int, cols: int, json_data: str) -> Tuple[List, str]:
    """Update the data preview based on user-specified row and column limits."""
    if not n_clicks or not json_data:
        raise PreventUpdate
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        rows = max(1, min(rows, len(df))) if rows else 10
        cols = max(1, min(cols, len(df.columns))) if cols else 10
        
        data_store['row_limit'] = rows
        data_store['col_limit'] = cols
        
        preview_df = df.head(rows).iloc[:, :cols]
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            style={'backgroundColor': 'white'}
        )
        
        return [
            [
                preview_table,
                html.Div(
                    f"Using {len(preview_df)} of {len(df)} rows and {len(preview_df.columns)} of {len(df.columns)} columns",
                    className="mt-2",
                    style={'color': COLORS['text_secondary']}
                )
            ],
            json.dumps(data_store)
        ]
        
    except Exception as e:
        logger.error(f"Preview update error: {str(e)}")
        return html.Div(f'Error updating preview: {str(e)}', style={'color': COLORS['error']}), no_update

# File Change Management
@app.callback(
    [Output('upload-data', 'style', allow_duplicate=True),
     Output('data-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children', allow_duplicate=True),
     Output('analyze-button', 'disabled', allow_duplicate=True),
     Output('viz-state', 'data', allow_duplicate=True)],
    Input('change-file-button', 'n_clicks'),
    State('upload-data', 'style'),
    prevent_initial_call=True
)
def change_file(n_clicks: int, current_style: Dict) -> Tuple:
    """Handle file change request."""
    if n_clicks:
        visible_style = {**current_style, 'display': 'block'}
        return visible_style, None, '', True, None
    return no_update, no_update, no_update, no_update, no_update

# Visualization Specs Import
@app.callback(
    [Output('viz-specs-modal', 'is_open'),
     Output('viz-specs-modal-content', 'children')],
    [Input('import-viz-specs-button', 'n_clicks'),
     Input('close-viz-specs-modal', 'n_clicks')],
    [State('viz-specs-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_viz_specs_modal(import_clicks: Optional[int], close_clicks: Optional[int], 
                         is_open: bool) -> Tuple[bool, Optional[html.Div]]:
    """Toggle and populate the visualization specifications import modal."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'import-viz-specs-button':
        viz_specs_dir = os.path.join(BASE_DIR, 'llm_responses')
        viz_specs_files = glob.glob(os.path.join(viz_specs_dir, 'viz_specs_*.json'))
        
        if not viz_specs_files:
            return True, html.Div("No visualization specifications found", className="text-muted")
        
        file_list = []
        for file_path in viz_specs_files:
            try:
                with open(file_path, 'r') as f:
                    specs = json.load(f)
                    timestamp_str = specs.get('timestamp', 'Unknown')
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_timestamp = timestamp_str
                        timestamp = datetime.min
                    
                    file_list.append({
                        'path': file_path,
                        'timestamp': timestamp,
                        'display_data': {
                            'timestamp': formatted_timestamp,
                            'model': specs.get('model', 'Unknown'),
                            'provider': specs.get('provider', 'Unknown'),
                            'dataset_filename': specs.get('dataset_filename', 'Unknown dataset')
                        }
                    })
            except Exception as e:
                logger.error(f"Error reading viz specs file {file_path}: {str(e)}")
                continue
        
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        list_items = [
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H6(f"Generated: {item['display_data']['timestamp']}", className="mb-1"),
                            html.Small([
                                f"Model: {item['display_data']['model']} ({item['display_data']['provider']})",
                                html.Br(),
                                f"For dataset: {item['display_data']['dataset_filename']}"
                            ], className="text-muted")
                        ]
                    ),
                    dbc.Button(
                        "Use",
                        id={'type': 'use-viz-specs', 'index': item['path']},
                        color="primary",
                        size="sm",
                        className="ms-auto"
                    )
                ],
                className="d-flex justify-content-between align-items-center"
            ) for item in file_list
        ]
        
        return True, dbc.ListGroup(list_items)
    
    return False, None

# Visualization Specs Usage
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('analyze-button', 'n_clicks', allow_duplicate=True),
     Output('viz-specs-modal', 'is_open', allow_duplicate=True),
     Output('viz-state', 'data')],
    Input({'type': 'use-viz-specs', 'index': ALL}, 'n_clicks'),
    [State('data-store', 'data'),
     State('analyze-button', 'n_clicks')],
    prevent_initial_call=True
)
def use_viz_specs(n_clicks: List[Optional[int]], current_data: str, 
                 current_clicks: Optional[int]) -> Tuple:
    """Import and apply previously saved visualization specifications."""
    ctx = dash.callback_context
    if not any(n_clicks):
        raise PreventUpdate
        
    try:
        triggered = ctx.triggered[0]
        file_path = eval(triggered['prop_id'].split('.')[0]+'"}')['index']
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Visualization specs file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            viz_specs = json.load(f)
        
        current_data = json.loads(current_data)
        current_data['imported_viz_specs'] = viz_specs['visualization_specs']
        
        return json.dumps(current_data), (current_clicks or 0) + 1, False, True
        
    except Exception as e:
        logger.error(f"Error in use_viz_specs: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise PreventUpdate

# Data Analysis Pipeline
@app.long_callback(
    [Output('results-container', 'children'),
     Output('dashboard-rendered', 'data'),
     Output('data-store', 'data', allow_duplicate=True)],
    Input('analyze-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('llm-provider', 'value'),
     State('api-key-input', 'value'),
     State('model-selection', 'value'),
     State('viz-only-checkbox', 'value'),
     State('kpi-selector', 'value')],
    prevent_initial_call=True,
    running=[
        (Output('analyze-button', 'disabled'), True, False),
        (Output('upload-data', 'disabled'), True, False),
        (Output('llm-provider', 'disabled'), True, False),
        (Output('api-key-input', 'disabled'), True, False),
        (Output('model-selection', 'disabled'), True, False),
        (Output('kpi-selector', 'disabled'), True, False),
    ],
    progress=[Output('upload-status', 'children')],
)
def analyze_data(set_progress, n_clicks: int, json_data: str, provider: str, 
                input_api_key: str, model: str, viz_only: bool, 
                kpis: List[str]) -> Tuple[html.Div, bool, str]:
    """Process the uploaded dataset and generate visualizations and analysis."""
    if not n_clicks or not json_data:
        raise PreventUpdate
    
    try:
        api_key = get_api_key(model) or input_api_key
        
        data_store = json.loads(json_data)
        df_full = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        filename = data_store.get('filename', 'unknown_file')
        
        df = df_full.head(data_store['row_limit']).iloc[:, :data_store['col_limit']]
        imported_viz_specs = data_store.get('imported_viz_specs')
        
        if imported_viz_specs:
            set_progress(html.Div("Using imported visualization specifications...", 
                                style={'color': COLORS['info']}))
            viz_specs = imported_viz_specs
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            analysis = "Analysis skipped - using imported visualization specifications"
            summary = "Summary skipped - using imported visualization specifications"
            
        else:
            if provider == 'external' and not api_key:
                raise ValueError("API key is required for external provider")
            
            set_progress(html.Div("Initializing analysis pipeline...", 
                                style={'color': COLORS['info']}))
            
            if provider == 'local':
                pipeline = LLMPipeline(model_name="llama3.1", use_local=True)
            else:
                os.environ["LLM_API_KEY"] = api_key
                pipeline = LLMPipeline(model_name=model, use_local=False)
            
            if not viz_only:
                set_progress(html.Div("1/5 Analyzing dataset... (Rate limiting in effect)", 
                                    style={'color': COLORS['info']}))
                analysis = pipeline.analyze_dataset(df, kpis)
            else:
                analysis = "Analysis skipped - visualizations only mode"
            
            set_progress(html.Div("2/5 Generating visualization suggestions... (Rate limiting in effect)", 
                                style={'color': COLORS['info']}))
            viz_specs = pipeline.suggest_visualizations(df, kpis, filename=filename)
            
            data_store['visualization_specs'] = viz_specs
            
            set_progress(html.Div("3/5 Creating visualizations...", 
                                style={'color': COLORS['info']}))
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            if not viz_only:
                set_progress(html.Div("4/5 Generating insights summary... (Rate limiting in effect)", 
                                    style={'color': COLORS['info']}))
                summary = pipeline.summarize_analysis(analysis, viz_specs)
            else:
                summary = "Summary skipped - visualizations only mode"
        
        set_progress(html.Div("5/5 Rendering dashboard...", 
                            style={'color': COLORS['info']}))
        
        components = [
            dbc.Card([
                dbc.CardHeader(
                    html.H3("Dashboard", 
                           className="mb-0",
                           style={
                               'color': COLORS['text_primary'],
                               'fontWeight': '500',
                               'fontSize': '1.5rem'
                           }
                    )
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(
                                    dbc.Tabs([
                                        dbc.Tab(
                                            label="Chart", 
                                            tab_id=f"chart-tab-{i}",
                                            label_style={
                                                'color': COLORS['text_primary']
                                            },
                                            active_label_style={
                                                'backgroundColor': COLORS['background'],
                                                'color': COLORS['primary']
                                            }
                                        ),
                                        dbc.Tab(
                                            label="Code", 
                                            tab_id=f"code-tab-{i}",
                                            label_style={
                                                'color': COLORS['text_primary']
                                            },
                                            active_label_style={
                                                'backgroundColor': COLORS['background'],
                                                'color': COLORS['primary']
                                            }
                                        )
                                    ],
                                    id={'type': 'tabs', 'index': i},
                                    active_tab=f"chart-tab-{i}",
                                    style={
                                        'borderBottom': 'none'
                                    }),
                                    style={
                                        'backgroundColor': COLORS['surface'],
                                        'borderBottom': f'1px solid {COLORS["divider"]}'
                                    }
                                ),
                                dbc.CardBody([
                                    html.Div([
                                        html.Div(
                                            dcc.Graph(
                                                id={'type': 'viz', 'index': i},
                                                figure=fig,
                                                config={'displayModeBar': False}
                                            ),
                                            id={'type': 'chart-content', 'index': i},
                                            style={
                                                'display': 'block',
                                                'backgroundColor': COLORS['plot_container'],
                                                'padding': '1rem',
                                                'borderRadius': '5px',
                                                'border': f'1px solid {COLORS["divider"]}'
                                            }
                                        ),
                                        html.Div([
                                            html.Pre(
                                                code,
                                                style={
                                                    'backgroundColor': COLORS['background'],
                                                    'padding': '1rem',
                                                    'borderRadius': '5px',
                                                    'whiteSpace': 'pre-wrap',
                                                    'fontSize': '0.8rem',
                                                    'color': COLORS['text_primary'],
                                                    'border': f'1px solid {COLORS["divider"]}'
                                                }
                                            ),
                                            dbc.Button(
                                                "Copy Code",
                                                id={'type': 'copy-btn', 'index': i},
                                                color="primary",
                                                size="sm",
                                                className="mt-2",
                                                style={
                                                    'backgroundColor': COLORS['primary'],
                                                    'borderColor': COLORS['primary'],
                                                    'boxShadow': f'0 2px 4px {COLORS["shadow"]}'
                                                }
                                            )
                                        ],
                                        id={'type': 'code-content', 'index': i},
                                        style={'display': 'none'}
                                        )
                                    ])
                                ],
                                style={'backgroundColor': COLORS['surface']})
                            ],
                            className="mb-4",
                            style={
                                'backgroundColor': COLORS['surface'],
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 8px rgba(0, 35, 76, 0.15), 0 1px 3px rgba(0, 35, 76, 0.1)',
                                'border': 'none',
                                'transition': 'box-shadow 0.3s ease'
                            })
                        ], xs=12, md=6) for i, (fig, code) in enumerate(figures.values())
                    ])
                ],
                style={'backgroundColor': COLORS['surface']})
            ],
            className='mb-4',
            style={
                'backgroundColor': COLORS['surface'],
                'borderRadius': '8px',
                'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                'border': 'none'
            })
        ]
        
        if not viz_only and not imported_viz_specs:
            components.extend([
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Key Insights (experimental)",
                               className="mb-0",
                               style={
                                   'color': COLORS['text_primary'],
                                   'fontWeight': '500',
                                   'fontSize': '1.5rem'
                               })
                    ),
                    dbc.CardBody(
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            markdown2.markdown(summary, extras=['tables', 'break-on-newline', 'cuddled-lists'])
                        ),
                        style={
                            'backgroundColor': COLORS['surface'],
                            'padding': '1rem',
                            'borderRadius': '5px',
                            'color': COLORS['text_primary']
                        }
                    )
                ],
                className='mb-4',
                style={
                    'backgroundColor': COLORS['surface'],
                    'borderRadius': '8px',
                    'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                    'border': 'none'
                }),
                
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Dataset Analysis (experimental)",
                               className="mb-0",
                               style={
                                   'color': COLORS['text_primary'],
                                   'fontWeight': '500',
                                   'fontSize': '1.5rem'
                               })
                    ),
                    dbc.CardBody(
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            markdown2.markdown(analysis, extras=['tables', 'break-on-newline', 'cuddled-lists'])
                        ),
                        style={
                            'backgroundColor': COLORS['surface'],
                            'padding': '1rem',
                            'borderRadius': '5px',
                            'color': COLORS['text_primary']
                        }
                    )
                ],
                style={
                    'backgroundColor': COLORS['surface'],
                    'borderRadius': '8px',
                    'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                    'border': 'none'
                })
            ])
        
        return html.Div(components), True, json.dumps(data_store)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return html.Div(
            f"Error during analysis: {str(e)}", 
            style={
                'color': COLORS['error'],
                'padding': '1rem',
                'backgroundColor': COLORS['surface'],
                'borderRadius': '8px',
                'boxShadow': f'0 2px 4px {COLORS["shadow"]}',
                'border': 'none'
            }
        ), False, json_data

# Tab Switching
@app.callback(
    [Output({'type': 'chart-content', 'index': MATCH}, 'style'),
     Output({'type': 'code-content', 'index': MATCH}, 'style')],
    Input({'type': 'tabs', 'index': MATCH}, 'active_tab'),
    prevent_initial_call=True
)
def switch_tab(active_tab: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Switch between chart and code views in visualization tabs."""
    if active_tab and 'chart-tab' in active_tab:
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

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
    Output({'type': 'copy-btn', 'index': MATCH}, 'children'),
    Input({'type': 'copy-btn', 'index': MATCH}, 'n_clicks'),
    State({'type': 'code-content', 'index': MATCH}, 'children'),
    prevent_initial_call=True
)

# Preview Visibility
@app.callback(
    Output('preview-section', 'style'),
    [Input('viz-state', 'data'),
     Input('change-file-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_preview_visibility(viz_active: bool, change_clicks: int) -> Dict[str, str]:
    """Toggle visibility of the data preview section."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'viz-state' and viz_active:
        return {'display': 'none'}
    elif trigger_id == 'change-file-button':
        return {'display': 'block'}
    
    return dash.no_update

# Button Text Update
@app.callback(
    Output('analyze-button', 'children'),
    Input('dashboard-rendered', 'data')
)
def update_button_text(dashboard_rendered: bool) -> str:
    """Update the analyze button text based on dashboard state."""
    return 'Regenerate Dashboard' if dashboard_rendered else 'Analyze Data'

# Filter Controls Creation
@app.callback(
    [Output('filter-controls', 'children'),
     Output('sidebar-collapse', 'is_open')],
    [Input('dashboard-rendered', 'data'),
     Input('data-store', 'data')],
    prevent_initial_call=True
)
def create_filter_controls(dashboard_rendered: bool, json_data: str) -> Tuple[List, bool]:
    """Create filter controls based on the dataset columns."""
    if not dashboard_rendered or not json_data:
        return [], False
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        filters = []
        temporal_col = None
        
        # Find temporal column
        for col in df.columns:
            try:
                temp_series = pd.to_datetime(df[col], errors='coerce')
                if not temp_series.isna().all():
                    temporal_col = col
                    df[col] = temp_series
                    break
            except:
                continue
        
        # Add temporal filter if found
        if temporal_col:
            min_date = df[temporal_col].min()
            max_date = df[temporal_col].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                min_date_str = min_date.strftime('%Y-%m-%d')
                max_date_str = max_date.strftime('%Y-%m-%d')
                
                filters.extend([
                    html.H6("Time Range", className="mt-3"),
                    dcc.DatePickerRange(
                        id='date-range-filter',
                        min_date_allowed=min_date_str,
                        max_date_allowed=max_date_str,
                        start_date=min_date_str,
                        end_date=max_date_str,
                        className="mb-3 w-100"
                    )
                ])
        
        # Add categorical filters
        categorical_cols = [
            col for col in df.columns 
            if col != temporal_col and df[col].nunique() / len(df) < 0.05
        ]
        
        for col in categorical_cols:
            unique_values = sorted(df[col].dropna().unique())
            if len(unique_values) > 0:
                filters.extend([
                    html.H6(f"{col}", className="mt-3"),
                    dcc.Dropdown(
                        id={'type': 'category-filter', 'column': col},
                        options=[{'label': str(val), 'value': str(val)} for val in unique_values],
                        value=[],
                        multi=True,
                        placeholder=f"Select {col}...",
                        className="mb-3"
                    )
                ])
        
        if filters:
            filters.append(
                dbc.Button(
                    "Reset Filters",
                    id="reset-filters-button",
                    color="secondary",
                    size="sm",
                    className="mt-3 w-100"
                )
            )
            
            return html.Div(filters, style={'padding': '10px'}), True
        
        return [], False
        
    except Exception as e:
        logger.error(f"Error creating filters: {str(e)}")
        return [], False

# Filter State Management
@app.callback(
    Output('filter-state', 'data'),
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input({'type': 'category-filter', 'column': ALL}, 'value'),
     Input({'type': 'category-filter', 'column': ALL}, 'id'),
     Input('reset-filters-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_filter_state(start_date: str, end_date: str, category_values: List, 
                       category_ids: List, reset_clicks: int) -> Optional[Dict]:
    """Update the filter state based on user selections."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    if ctx.triggered[0]['prop_id'] == 'reset-filters-button.n_clicks':
        return None
        
    filter_state = {
        'temporal': {
            'start_date': start_date,
            'end_date': end_date
        },
        'categorical': {
            id['column']: values for id, values in zip(category_ids, category_values)
            if values
        }
    }
    
    return filter_state

# Visualization Updates
@app.callback(
    [Output({'type': 'viz', 'index': ALL}, 'figure'),
     Output('date-range-filter', 'start_date'),
     Output('date-range-filter', 'end_date'),
     Output({'type': 'category-filter', 'column': ALL}, 'value')],
    [Input('reset-filters-button', 'n_clicks'),
     Input('filter-state', 'data')],
    [State('data-store', 'data'),
     State({'type': 'viz', 'index': ALL}, 'figure'),
     State({'type': 'category-filter', 'column': ALL}, 'id')],
    prevent_initial_call=True
)
def update_visualizations(reset_clicks: int, filter_state: Dict, json_data: str, 
                        current_figures: List, category_ids: List) -> Tuple:
    """Update all visualizations based on the current filter state."""
    ctx = dash.callback_context
    if not ctx.triggered or not json_data:
        raise PreventUpdate
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        viz_specs = data_store.get('imported_viz_specs') or data_store.get('visualization_specs')
        if not viz_specs:
            logger.warning("No visualization specifications found in data store")
            return current_figures, None, None, [[] for _ in category_ids]
        
        if ctx.triggered[0]['prop_id'] == 'reset-filters-button.n_clicks':
            logger.info("Resetting all filters")
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            new_figures = [
                list(figures.values())[i][0] if i < len(figures) else current_figures[i]
                for i in range(len(current_figures))
            ]
            
            temporal_col = None
            min_date = None
            max_date = None
            for col in df.columns:
                try:
                    temp_series = pd.to_datetime(df[col], errors='coerce')
                    if not temp_series.isna().all():
                        temporal_col = col
                        df[col] = temp_series
                        min_date = df[col].min().strftime('%Y-%m-%d')
                        max_date = df[col].max().strftime('%Y-%m-%d')
                        break
                except:
                    continue
            
            return new_figures, min_date, max_date, [[] for _ in category_ids]
        
        if filter_state:
            filtered_df = apply_filters(df, filter_state)
            
            dashboard_builder = DashboardBuilder(filtered_df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            new_figures = [
                list(figures.values())[i][0] if i < len(figures) else current_figures[i]
                for i in range(len(current_figures))
            ]
            
            logger.info(f"Created {len(new_figures)} new figures with filtered data")
            
            return (
                new_figures,
                filter_state['temporal'].get('start_date'),
                filter_state['temporal'].get('end_date'),
                [filter_state['categorical'].get(cat_id['column'], []) for cat_id in category_ids]
            )
        
        return current_figures, None, None, [[] for _ in category_ids]
        
    except Exception as e:
        logger.error(f"Error updating visualizations: {str(e)}")
        return current_figures, None, None, [[] for _ in category_ids]

# KPI Selector Population
@app.callback(
    Output('kpi-selector', 'options'),
    Input('data-store', 'data'),
    prevent_initial_call=True
)
def update_kpi_selector(json_data: str) -> List[Dict[str, str]]:
    """Update KPI selector options based on loaded dataset columns."""
    if not json_data:
        return []
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        return [{'label': col, 'value': col} for col in df.columns]
        
    except Exception as e:
        logger.error(f"Error updating KPI selector: {str(e)}")
        return []

# --- 8. MAIN EXECUTION ---
if __name__ == '__main__':
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )

