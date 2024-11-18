"""
AI Dashboard Builder - A Dash application for automated dashboard creation using LLMs.

This module serves as the main entry point for the AI Dashboard Builder application.
It handles the web interface, data processing, and visualization generation using
various LLM providers.
"""

import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, no_update, long_callback, MATCH, ALL
import dash
from llm_pipeline import LLMPipeline
from dashboard_builder import DashboardBuilder
import base64
import io
import json
import logging
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import os
from io import BytesIO
import glob
from datetime import datetime
from dotenv import load_dotenv
import dash_dangerously_set_inner_html
import markdown2
import plotly.graph_objects as go
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Dashboard')
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Setup diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Color Palette - Salmon theme
COLORS = {
    'background': "#FFF5F5",
    'card': "#FFF0F0",
    'divider': "#FFD7D7",
    'primary': "#FF9999",
    'secondary': "#FF7777",
    'warning': "#FFB366",
    'error': "#FF6B6B",
    'text_primary': "#4A4A4A",
    'text_secondary': "#717171",
    'highlight': "#FF8585",
    'info': "#85A3FF"
}

# Add these constants near the top of the file, after COLORS
MAX_PREVIEW_ROWS = 1000  # Default maximum rows to show in preview
MAX_PREVIEW_COLS = 20    # Default maximum columns to show in preview

# Define the base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv()

# Function to get the appropriate API key based on model
def get_api_key(model_name: str) -> str:
    """
    Get the appropriate API key based on the model name.
    
    Args:
        model_name (str): Name of the LLM model (e.g., 'gpt', 'claude', 'mistral')
        
    Returns:
        str: API key for the specified model, or empty string if not found
    """
    if 'gpt' in model_name:
        return os.getenv('OPENAI_API_KEY', '')
    elif 'claude' in model_name:
        return os.getenv('ANTHROPIC_API_KEY', '')
    elif 'mistral' in model_name:
        return os.getenv('MISTRAL_API_KEY', '')
    elif 'mixtral' in model_name or 'groq' in model_name or 'llama' in model_name or 'gemma' in model_name:
        return os.getenv('GROQ_API_KEY', '')
    return ''

# Initialize app with long callback manager and title
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=False,
    update_title=None,
    long_callback_manager=long_callback_manager,
    title="AI Dashboard Builder"
)

# Set the title for the browser tab
app.title = "AI Dashboard Builder"
# Modify the layout for responsive design
app.layout = html.Div([
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='viz-state', storage_type='memory'),
    dcc.Store(id='dashboard-rendered', storage_type='memory'),
    dcc.Store(id='filter-state', storage_type='memory'),  # For storing filter state
    
    # Main container
    dbc.Container(fluid=True, children=[
        html.A(
            html.H1('AI Dashboard Builder',
                style={
                    'textAlign': 'center',
                    'color': COLORS['primary'],
                    'marginBottom': '2rem',
                    'paddingTop': '1rem',
                    'textDecoration': 'none'
                }
            ),
            href='/',
            style={'textDecoration': 'none'}
        ),
        
        # Controls Row - Horizontal bar at the top
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # LLM Provider Column
                    dbc.Col([
                        html.H5("LLM Provider", className="mb-2"),
                        dbc.RadioItems(
                            id='llm-provider',
                            options=[
                                {'label': ['Local ', html.A('(Ollama)', href='https://ollama.com/download', target='_blank')], 'value': 'local'},
                                {'label': 'External API', 'value': 'external'}
                            ],
                            value='local',
                            className="mb-2",
                            inline=True
                        ),
                        dbc.Collapse(
                            dbc.Input(
                                id='api-key-input',
                                type='password',
                                placeholder='Enter API Key',
                                className="mb-2"
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
                    
                    # File Upload Column
                    dbc.Col([
                        html.H5("Dataset Upload", className="mb-2"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select a CSV/Excel File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'backgroundColor': COLORS['background']
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mt-2"),
                    ], xs=12, md=5),
                    
                    # Analyze Button Column
                    dbc.Col([
                        html.H5("\u00A0", className="mb-2"),  # Invisible header for alignment
                        dbc.Button(
                            'Analyze Data',
                            id='analyze-button',
                            color='primary',
                            className='w-100 mt-2',
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
                    ], xs=12, md=3, className="d-flex align-items-end flex-column"),
                ])
            ])
        ], className="mb-4"),
        
        # Results Section with Sidebar and Content
        dbc.Row([
            # Collapsible Sidebar - Make it narrower and add margin
            dbc.Col([
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Inferred Data Filters"),
                        dbc.CardBody(id='filter-controls')
                    ], className="sticky-top"),  # Make filters stick while scrolling
                    id="sidebar-collapse",
                    is_open=False,
                ),
            ], width=2, style={'paddingRight': '20px'}),  # Reduced width and added right padding
            
            # Main Content - Give it more space
            dbc.Col([
                dbc.Spinner(
                    html.Div(
                        id='results-container',
                        style={'minHeight': '200px'}
                    ),
                    color='primary',
                    type='border',
                    fullscreen=False,
                )
            ], width=10)  # Increased width
        ], className="g-0")  # Remove default gutters
    ])
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# Callback to toggle API key and model selection visibility
@app.callback(
    [Output('api-key-collapse', 'is_open'),
     Output('model-selection-collapse', 'is_open'),
     Output('api-key-input', 'value'),
     Output('api-key-input', 'placeholder')],
    [Input('llm-provider', 'value'),
     Input('model-selection', 'value')]
)
def toggle_api_key(provider: str, model: str) -> tuple:
    """
    Toggle visibility and populate API key input based on provider selection.
    
    Args:
        provider (str): Selected provider ('local' or 'external')
        model (str): Selected model name
        
    Returns:
        tuple: (
            bool: Whether to show API key input,
            bool: Whether to show model selection,
            str: API key value if available,
            str: Placeholder text for API key input
        )
    """
    if provider != 'external':
        return False, False, '', 'Enter API Key'
    
    api_key = get_api_key(model)
    if api_key:
        return True, True, api_key, 'API KEY loaded'
    else:
        return True, True, '', 'Enter API Key'

# Add this helper function after the imports and before the app initialization
def smart_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently convert string columns to numeric types where possible.
    Handles various number formats, currencies, percentages, and is fault-tolerant.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with converted numeric columns
    """
    def clean_numeric_string(s):
        if pd.isna(s):
            return s
        if not isinstance(s, str):
            return s
            
        # Remove currency symbols, parentheses for negative numbers, and other common characters
        s = str(s).strip()
        s = re.sub(r'[($€£¥,)]', '', s)
        
        # Handle parentheses negative numbers e.g., "(123)" -> "-123"
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
            
        # Handle percentages
        if s.endswith('%'):
            try:
                return float(s.rstrip('%')) / 100
            except:
                return s
                
        # Handle thousands/millions/billions indicators
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        if s and s[-1].upper() in multipliers:
            try:
                return float(s[:-1]) * multipliers[s[-1].upper()]
            except:
                return s
                
        return s

    def try_numeric_conversion(series):
        # First clean the strings
        cleaned = series.map(clean_numeric_string)
        
        # Try converting to numeric
        try:
            numeric = pd.to_numeric(cleaned, errors='coerce')
            # Only convert if we don't lose too many values
            na_ratio = numeric.isna().sum() / len(numeric)
            if na_ratio < 0.3:  # Allow up to 30% NaN values
                return numeric
        except:
            pass
        
        return series

    # Process each column
    df_converted = df.copy()
    for col in df.columns:
        # Skip if already numeric
        if np.issubdtype(df[col].dtype, np.number):
            continue
            
        # Skip if datetime
        try:
            pd.to_datetime(df[col], errors='raise')
            continue
        except:
            pass
            
        # Try converting to numeric
        df_converted[col] = try_numeric_conversion(df[col])

    logger.info(f"Completed smart numeric conversion: {df_converted.head()}")
        
    return df_converted

# Modify the file upload callback
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
def handle_upload(contents, filename, current_style):
    """
    Process uploaded data file and prepare it for analysis.
    
    Handles CSV and Excel file uploads, creates a preview table,
    and stores the data for analysis.
    
    Args:
        contents (str): Base64 encoded file contents
        filename (str): Name of uploaded file
        current_style (dict): Current upload component style
        
    Returns:
        tuple: (
            str: JSON string of stored data,
            html.Div: Upload status and preview components,
            bool: Whether analyze button should be enabled,
            dict: Updated upload component style
        )
        
    Raises:
        Exception: If file reading or processing fails
    """
    if contents is None:
        return no_update, no_update, True, current_style
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Check file extension
        file_extension = filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            return None, html.Div('Please upload a CSV or Excel file', style={'color': COLORS['error']}), True, current_style
            
        # Read the file based on its extension
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:  # Excel files
            df = pd.read_excel(BytesIO(decoded))
            
        if df.empty:
            return None, html.Div('The uploaded file is empty', style={'color': COLORS['error']}), True, current_style
        
        # Apply smart numeric conversion
        df = smart_numeric_conversion(df)
        
        # Create preview controls with max values by default
        preview_controls = dbc.Row([
            dbc.Col([
                html.Label("Limit Rows:", className="me-2"),
                dbc.Input(
                    id='preview-rows-input',
                    type='number',
                    min=1,
                    max=len(df),
                    value=len(df),  # Default to 10 rows
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
                    value=len(df.columns),  # Show all columns by default
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

        # Create initial preview table with limited rows
        preview_df = df.head(10)  # Only show first 10 rows
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            style={'backgroundColor': 'white'}
        )
            
        # Store the full dataframe but with initial preview limits
        data_store = {
            'full_data': df.to_json(date_format='iso', orient='split'),
            'row_limit': 10,  # Initial limit of 10 rows
            'col_limit': len(df.columns)  # Show all columns initially
        }
        
        # Hide the upload component after successful upload
        hidden_style = {**current_style, 'display': 'none'}

        # Add import viz specs button
        import_button = html.Div([
            dbc.Button(
                [
                    html.I(className="fas fa-file-import me-2"),  # Font Awesome import icon
                    "Import Previous Viz Specs"
                ],
                id='import-viz-specs-button',
                color="link",  # Makes it look like a link
                className="p-0",  # Remove padding
                style={
                    'color': '#6c757d',  # Muted color
                    'fontSize': '0.8rem',  # Smaller text
                    'textDecoration': 'none',  # No underline
                    'opacity': '0.7'  # Slightly faded
                }
            ),
            dbc.Tooltip(
                "Developer option: Reuse previously generated visualization specifications",
                target='import-viz-specs-button',
                placement='right'
            ),
            # Hidden div to store available viz specs files
            html.Div(id='viz-specs-list', style={'display': 'none'}),
            # Modal for selecting viz specs file
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
                # File info and buttons (always visible)
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
                
                # Preview section (can be hidden)
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

# Modify the update preview callback to also update the stored limits
@app.callback(
    [Output('preview-table-container', 'children'),
     Output('data-store', 'data', allow_duplicate=True)],
    [Input('update-preview-button', 'n_clicks')],
    [State('preview-rows-input', 'value'),
     State('preview-cols-input', 'value'),
     State('data-store', 'data')],
    prevent_initial_call=True
)
def update_preview(n_clicks, rows, cols, json_data):
    """
    Update the data preview based on user-specified row and column limits.
    
    Args:
        n_clicks (int): Number of update button clicks
        rows (int): Number of rows to show
        cols (int): Number of columns to show
        json_data (str): Current data store JSON string
        
    Returns:
        tuple: (
            list: Updated preview table components,
            str: Updated data store JSON with new limits
        )
        
    Raises:
        PreventUpdate: If button not clicked or no data
    """
    if not n_clicks or not json_data:
        raise PreventUpdate
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        # Ensure valid values
        rows = max(1, min(rows, len(df))) if rows else 10
        cols = max(1, min(cols, len(df.columns))) if cols else 10
        
        # Update the stored limits
        data_store['row_limit'] = rows
        data_store['col_limit'] = cols
        
        # Create preview with specified limits
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
            json.dumps(data_store)  # Update stored limits
        ]
        
    except Exception as e:
        logger.error(f"Preview update error: {str(e)}", exc_info=True)
        return html.Div(f'Error updating preview: {str(e)}', style={'color': COLORS['error']}), no_update

# Add callback for the change file button
@app.callback(
    [Output('upload-data', 'style', allow_duplicate=True),
     Output('data-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children', allow_duplicate=True),
     Output('analyze-button', 'disabled', allow_duplicate=True),
     Output('viz-state', 'data', allow_duplicate=True)],  # Add this output
    Input('change-file-button', 'n_clicks'),
    State('upload-data', 'style'),
    prevent_initial_call=True
)
def change_file(n_clicks, current_style):
    if n_clicks:
        # Show the upload component again
        visible_style = {**current_style, 'display': 'block'}
        return visible_style, None, '', True, None  # Reset viz-state to None
    return no_update, no_update, no_update, no_update, no_update

# Add callbacks for the viz specs import functionality
@app.callback(
    [Output('viz-specs-modal', 'is_open'),
     Output('viz-specs-modal-content', 'children')],
    [Input('import-viz-specs-button', 'n_clicks'),
     Input('close-viz-specs-modal', 'n_clicks')],
    [State('viz-specs-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_viz_specs_modal(import_clicks, close_clicks, is_open):
    """
    Toggle and populate the visualization specifications import modal.
    
    Args:
        import_clicks (int): Number of clicks on import button
        close_clicks (int): Number of clicks on close button
        is_open (bool): Current modal state
        
    Returns:
        tuple: (
            bool: Whether modal should be open,
            html.Div: Modal content if opening
        )
        
    Note:
        When opening, scans for and lists available visualization specification files
        sorted by most recent first.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'import-viz-specs-button':
        # Use absolute path for viz specs directory
        viz_specs_dir = os.path.join(BASE_DIR, 'llm_responses')
        viz_specs_files = glob.glob(os.path.join(viz_specs_dir, 'viz_specs_*.json'))
        
        if not viz_specs_files:
            return True, html.Div("No visualization specifications found", className="text-muted")
        
        # Create list of files with metadata
        file_list = []
        for file_path in viz_specs_files:
            try:
                with open(file_path, 'r') as f:
                    specs = json.load(f)
                    # Parse timestamp to datetime for sorting
                    timestamp_str = specs.get('timestamp', 'Unknown')
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_timestamp = timestamp_str
                        timestamp = datetime.min  # For sorting purposes
                    
                    model = specs.get('model', 'Unknown')
                    provider = specs.get('provider', 'Unknown')
                    
                    file_list.append({
                        'path': file_path,
                        'timestamp': timestamp,  # For sorting
                        'display_data': {
                            'timestamp': formatted_timestamp,
                            'model': model,
                            'provider': provider
                        }
                    })
            except Exception as e:
                logger.error(f"Error reading viz specs file {file_path}: {str(e)}")
                continue
        
        # Sort by timestamp, most recent first
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Create list items
        list_items = [
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H6(f"Generated: {item['display_data']['timestamp']}", className="mb-1"),
                            html.Small(
                                f"Model: {item['display_data']['model']} ({item['display_data']['provider']})", 
                                className="text-muted"
                            )
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

# Modify the use_viz_specs callback to include ctx
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('analyze-button', 'n_clicks', allow_duplicate=True),
     Output('viz-specs-modal', 'is_open', allow_duplicate=True),
     Output('viz-state', 'data')],  # Add this output
    Input({'type': 'use-viz-specs', 'index': ALL}, 'n_clicks'),
    [State('data-store', 'data'),
     State('analyze-button', 'n_clicks')],
    prevent_initial_call=True
)
def use_viz_specs(n_clicks, current_data, current_clicks):
    """
    Import and apply previously saved visualization specifications.
    
    Args:
        n_clicks (list): List of click counts for each import button
        current_data (str): Current data store JSON string
        current_clicks (int): Current analyze button click count
        
    Returns:
        tuple: (
            str: Updated data store JSON,
            int: Incremented analyze button clicks,
            bool: Whether to close the modal,
            bool: Whether visualizations are active
        )
        
    Raises:
        PreventUpdate: If no import button was clicked
    """
    ctx = dash.callback_context
    if not any(n_clicks):
        raise PreventUpdate
        
    try:
        # Get the triggered input info
        triggered = ctx.triggered[0]
        logger.info(f"Full triggered object: {triggered}")
        
        # Extract the file path from the triggered input ID and ensure it has .json extension
        file_path = eval(triggered['prop_id'].split('.')[0]+'"}')['index']
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        logger.info(f"File path with extension: {file_path}")
        
        # Verify file exists using absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            # Try to list existing files in the directory for debugging
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path):
                existing_files = os.listdir(dir_path)
                logger.info(f"Files in directory {dir_path}: {existing_files}")
            raise FileNotFoundError(f"Visualization specs file not found: {file_path}")
        
        # Load the selected viz specs
        with open(file_path, 'r') as f:
            viz_specs = json.load(f)
        logger.info(f"Successfully loaded viz specs from {file_path}")
        
        # Update the data store with the imported viz specs
        current_data = json.loads(current_data)
        current_data['imported_viz_specs'] = viz_specs['visualization_specs']
        
        # Create minimal upload status without preview
        upload_status = html.Div([
            html.Div(
                "Dataset loaded and visualization specs imported",
                style={'color': COLORS['info']}
            ),
            html.Button(
                'Change File',
                id='change-file-button',
                className='mt-2 mb-3 btn btn-outline-secondary btn-sm',
                n_clicks=0
            )
        ])
        
        # Return updated data store, increment analyze button clicks, close modal, and update upload status
        logger.info("Returning updated data store and incrementing clicks")
        return json.dumps(current_data), (current_clicks or 0) + 1, False, True
        
    except Exception as e:
        logger.error(f"Error in use_viz_specs: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise PreventUpdate

# Modify the analyze_data callback to include KPI information
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
     State('kpi-selector', 'value')],  # Add KPI selector state
    prevent_initial_call=True,
    running=[
        (Output('analyze-button', 'disabled'), True, False),
        (Output('upload-data', 'disabled'), True, False),
        (Output('llm-provider', 'disabled'), True, False),
        (Output('api-key-input', 'disabled'), True, False),
        (Output('model-selection', 'disabled'), True, False),
        (Output('kpi-selector', 'disabled'), True, False),  # Add KPI selector to disabled states
    ],
    progress=[Output('upload-status', 'children')],
)
def analyze_data(set_progress, n_clicks, json_data, provider, input_api_key, model, viz_only, kpis):
    """
    Process the uploaded dataset and generate visualizations and analysis.
    
    This function handles the main data analysis pipeline:
    1. Dataset analysis (if not viz_only)
    2. Visualization suggestion generation
    3. Dashboard creation
    4. Insights summary (if not viz_only)
    
    Args:
        set_progress: Callback to update progress status
        n_clicks (int): Number of button clicks
        json_data (str): JSON string containing dataset and metadata
        provider (str): Selected LLM provider ('local' or 'external')
        input_api_key (str): User-provided API key
        model (str): Selected model name
        viz_only (bool): Whether to skip analysis and only generate visualizations
        kpis (list): List of selected KPI columns
        
    Returns:
        tuple: (
            html.Div: Dashboard components,
            bool: Whether dashboard was successfully rendered
        )
        
    Raises:
        PreventUpdate: If no clicks or no data
        ValueError: If API key is missing for external provider
    """
    if not n_clicks or not json_data:
        raise PreventUpdate
    
    try:
        # Get API key from environment if available, otherwise use input
        api_key = get_api_key(model) or input_api_key
        
        data_store = json.loads(json_data)
        df_full = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        # Apply the stored limits to create the analysis dataset
        df = df_full.head(data_store['row_limit']).iloc[:, :data_store['col_limit']]
        
        # Check if we have imported viz specs
        imported_viz_specs = data_store.get('imported_viz_specs')
        
        if imported_viz_specs:
            set_progress(html.Div("Using imported visualization specifications...", style={'color': COLORS['info']}))
            viz_specs = imported_viz_specs
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            # Set dummy values for analysis and summary since we're using imported specs
            analysis = "Analysis skipped - using imported visualization specifications"
            summary = "Summary skipped - using imported visualization specifications"
            
        else:
            # Validate API key if using external provider
            if provider == 'external' and not api_key:
                raise ValueError("API key is required for external provider")
            
            set_progress(html.Div("Initializing analysis pipeline...", style={'color': COLORS['info']}))
            
            # Set up pipeline based on provider
            if provider == 'local':
                pipeline = LLMPipeline(model_name="llama3.1", use_local=True)
            else:
                # Set API key in environment
                os.environ["LLM_API_KEY"] = api_key
                pipeline = LLMPipeline(model_name=model, use_local=False)
            
            # Skip analysis if viz_only is True
            if not viz_only:
                set_progress(html.Div("1/5 Analyzing dataset... (Rate limiting in effect)", style={'color': COLORS['info']}))
                analysis = pipeline.analyze_dataset(df, kpis)  # Pass KPIs
            else:
                analysis = "Analysis skipped - visualizations only mode"
            
            set_progress(html.Div("2/5 Generating visualization suggestions... (Rate limiting in effect)", style={'color': COLORS['info']}))
            viz_specs = pipeline.suggest_visualizations(df, kpis)  # Pass KPIs
            
            # Store the viz specs in data_store
            data_store['visualization_specs'] = viz_specs
            
            set_progress(html.Div("3/5 Creating visualizations...", style={'color': COLORS['info']}))
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            # Skip summary if viz_only is True
            if not viz_only:
                set_progress(html.Div("4/5 Generating insights summary... (Rate limiting in effect)", style={'color': COLORS['info']}))
                summary = pipeline.summarize_analysis(analysis, viz_specs)
            else:
                summary = "Summary skipped - visualizations only mode"
        
        set_progress(html.Div("5/5 Rendering dashboard...", style={'color': COLORS['info']}))
        
        components = [
            # Visualizations Section (First)
            dbc.Card([
                dbc.CardHeader(
                    html.H3("Dashboard", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        # Create pairs of visualizations for two-column layout
                        *[
                            dbc.Col([
                                # Tabs for visualization
                                dbc.Card([
                                    dbc.CardHeader(
                                        dbc.Tabs([
                                            dbc.Tab(label="Chart", tab_id=f"chart-tab-{i}"),
                                            dbc.Tab(label="Code", tab_id=f"code-tab-{i}")
                                        ],
                                        id={'type': 'tabs', 'index': i},
                                        active_tab=f"chart-tab-{i}"),
                                    ),
                                    dbc.CardBody([
                                        html.Div([
                                            # Chart tab content
                                            html.Div(
                                                dcc.Graph(
                                                    id={'type': 'viz', 'index': i},
                                                    figure=fig,
                                                    config={'displayModeBar': False}
                                                ),
                                                id={'type': 'chart-content', 'index': i},
                                                style={'display': 'block'}
                                            ),
                                            # Code tab content
                                            html.Div([
                                                html.Pre(
                                                    code,
                                                    style={
                                                        'backgroundColor': COLORS['background'],
                                                        'padding': '1rem',
                                                        'borderRadius': '5px',
                                                        'whiteSpace': 'pre-wrap',
                                                        'fontSize': '0.8rem'
                                                    }
                                                ),
                                                dbc.Button(
                                                    "Copy Code",
                                                    id={'type': 'copy-btn', 'index': i},
                                                    color="primary",
                                                    size="sm",
                                                    className="mt-2"
                                                )
                                            ],
                                            id={'type': 'code-content', 'index': i},
                                            style={'display': 'none'}
                                            )
                                        ])
                                    ])
                                ], className="mb-4")
                            ], xs=12, md=6) for i, (fig, code) in enumerate(figures.values())
                        ]
                    ])
                ])
            ], className='mb-4'),
        ]
        
        # Only add Key Insights and Dataset Analysis if not viz_only
        if not viz_only and not imported_viz_specs:
            components.extend([
                # Key Insights Section
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Key Insights (experimental)", className="mb-0")
                    ),
                    dbc.CardBody(
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            markdown2.markdown(
                                summary,
                                extras=['tables', 'break-on-newline', 'cuddled-lists']
                            )
                        ),
                        style={
                            'backgroundColor': COLORS['background'],
                            'padding': '1rem',
                            'borderRadius': '5px',
                        }
                    )
                ], className='mb-4'),
                
                # Dataset Analysis Section
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Dataset Analysis (experimental)", className="mb-0")
                    ),
                    dbc.CardBody(
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            markdown2.markdown(
                                analysis,
                                extras=['tables', 'break-on-newline', 'cuddled-lists']
                            )
                        ),
                        style={
                            'backgroundColor': COLORS['background'],
                            'padding': '1rem',
                            'borderRadius': '5px'
                        }
                    )
                ])
            ])
        
        return html.Div(components), True, json.dumps(data_store)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return html.Div(
            f"Error during analysis: {str(e)}", 
            style={'color': COLORS['error'], 'padding': '1rem'}
        ), False, json_data  # Return original data_store on error

# Fix the tab switching callback
@app.callback(
    [Output({'type': 'chart-content', 'index': MATCH}, 'style'),
     Output({'type': 'code-content', 'index': MATCH}, 'style')],
    Input({'type': 'tabs', 'index': MATCH}, 'active_tab'),
    prevent_initial_call=True
)
def switch_tab(active_tab: str) -> tuple:
    """
    Switch between chart and code views in visualization tabs.
    
    Args:
        active_tab (str): ID of the active tab
        
    Returns:
        tuple: (
            dict: Style for chart content,
            dict: Style for code content
        )
    """
    if active_tab and 'chart-tab' in active_tab:
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

# Add callback for code copying
app.clientside_callback(
    """
    function(n_clicks, code_content) {
        if (n_clicks) {
            // Extract the code from the Pre component
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

# Modify the preview visibility callback to only target the preview section
@app.callback(
    Output('preview-section', 'style'),
    [Input('viz-state', 'data'),
     Input('change-file-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_preview_visibility(viz_active: bool, change_clicks: int) -> dict:
    """
    Toggle visibility of the data preview section.
    
    Args:
        viz_active (bool): Whether visualizations are currently active
        change_clicks (int): Number of clicks on change file button
        
    Returns:
        dict: CSS style for preview section
        
    Raises:
        PreventUpdate: If no trigger
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'viz-state' and viz_active:
        return {'display': 'none'}
    elif trigger_id == 'change-file-button':
        return {'display': 'block'}
    
    return dash.no_update

# Add a callback to update the button text
@app.callback(
    Output('analyze-button', 'children'),
    Input('dashboard-rendered', 'data')
)
def update_button_text(dashboard_rendered: bool) -> str:
    """
    Update the analyze button text based on dashboard state.
    
    Args:
        dashboard_rendered (bool): Whether a dashboard is currently rendered
        
    Returns:
        str: Button text
    """
    if dashboard_rendered:
        return 'Regenerate Dashboard'
    return 'Analyze Data'

# Add callback to create filter controls when dashboard is rendered
@app.callback(
    [Output('filter-controls', 'children'),
     Output('sidebar-collapse', 'is_open')],
    [Input('dashboard-rendered', 'data'),
     Input('data-store', 'data')],
    prevent_initial_call=True
)
def create_filter_controls(dashboard_rendered, json_data):
    """
    Create filter controls based on the dataset columns.
    """
    if not dashboard_rendered or not json_data:
        return [], False
        
    try:
        # Load the dataset
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        filters = []
        
        # Find temporal column
        temporal_col = None
        for col in df.columns:
            try:
                # Convert to datetime and check if conversion was successful
                temp_series = pd.to_datetime(df[col], errors='coerce')
                # Only consider as temporal if we have valid dates (not all NaT)
                if not temp_series.isna().all():
                    temporal_col = col
                    df[col] = temp_series  # Keep the converted dates
                    break
            except:
                continue
        
        # Add temporal filter if found
        if temporal_col:
            min_date = df[temporal_col].min()
            max_date = df[temporal_col].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                # Convert timestamps to strings for DatePickerRange
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
        categorical_cols = []
        for col in df.columns:
            if col != temporal_col:
                n_unique = df[col].nunique()
                if n_unique is not None and len(df) > 0:
                    if n_unique / len(df) < 0.05:  # Less than 5% unique values
                        categorical_cols.append(col)
        
        for col in categorical_cols:
            unique_values = sorted(df[col].dropna().unique())
            if len(unique_values) > 0:  # Only add filter if we have values
                filters.extend([
                    html.H6(f"{col}", className="mt-3"),
                    dcc.Dropdown(
                        id={'type': 'category-filter', 'column': col},
                        options=[{'label': str(val), 'value': str(val)} for val in unique_values],
                        value=[],  # Default to no selection (show all)
                        multi=True,
                        placeholder=f"Select {col}...",
                        className="mb-3"
                    )
                ])
        
        # Add reset button if we have any filters
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
            
            # Wrap all filters in a div with padding
            filters = html.Div(filters, style={'padding': '10px'})
            return filters, True
        
        return [], False
        
    except Exception as e:
        logger.error(f"Error creating filters: {str(e)}")
        return [], False

# Add callback to apply filters
@app.callback(
    Output('filter-state', 'data'),
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input({'type': 'category-filter', 'column': ALL}, 'value'),
     Input({'type': 'category-filter', 'column': ALL}, 'id'),
     Input('reset-filters-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_filter_state(start_date, end_date, category_values, category_ids, reset_clicks):
    """
    Update the filter state based on user selections.
    
    Returns:
        dict: Current state of all filters
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    # Handle reset
    if ctx.triggered[0]['prop_id'] == 'reset-filters-button.n_clicks':
        return None
        
    # Create filter state
    filter_state = {
        'temporal': {
            'start_date': start_date,
            'end_date': end_date
        },
        'categorical': {
            id['column']: values for id, values in zip(category_ids, category_values)
            if values  # Only include filters with selected values
        }
    }
    
    return filter_state

# Modify the existing callbacks to apply filters
def apply_filters(df: pd.DataFrame, filter_state: dict) -> pd.DataFrame:
    """
    Apply filters to the dataframe.
    
    Args:
        df (pd.DataFrame): Original dataframe
        filter_state (dict): Current filter state
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if not filter_state:
        return df
        
    filtered_df = df.copy()
    
    # Apply temporal filter
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
    
    # Apply categorical filters
    categorical_filters = filter_state.get('categorical', {})
    for col, values in categorical_filters.items():
        if values:  # Only apply filter if values are selected
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
    
    return filtered_df

# Modify the callback to update visualizations based on filters
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
def update_visualizations(reset_clicks, filter_state, json_data, current_figures, category_ids):
    """
    Update all visualizations based on the current filter state.
    """
    ctx = dash.callback_context
    if not ctx.triggered or not json_data:
        raise PreventUpdate
        
    try:
        # Load the dataset and viz specs
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        # Get viz specs from the correct location in data_store
        viz_specs = None
        if 'imported_viz_specs' in data_store:
            viz_specs = data_store['imported_viz_specs']
        elif 'visualization_specs' in data_store:
            viz_specs = data_store['visualization_specs']
            
        if not viz_specs:
            logger.warning("No visualization specifications found in data store")
            return current_figures, None, None, [[] for _ in category_ids]
        
        # If reset button clicked, reset all filters and recreate figures with original data
        if ctx.triggered[0]['prop_id'] == 'reset-filters-button.n_clicks':
            logger.info("Resetting all filters")
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            # Ensure we have the same number of figures as current_figures
            new_figures = []
            for i in range(len(current_figures)):
                if i < len(figures):
                    fig, _ = list(figures.values())[i]
                    new_figures.append(fig)
                else:
                    new_figures.append(current_figures[i])
            
            # Find temporal column and its min/max dates
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
            
            # Return reset state
            return new_figures, min_date, max_date, [[] for _ in category_ids]
        
        # Handle normal filter updates
        if filter_state:
            filtered_df = df.copy()
            # Apply temporal filter
            temporal_filter = filter_state.get('temporal', {})
            if temporal_filter.get('start_date') and temporal_filter.get('end_date'):
                for col in df.columns:
                    try:
                        filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')
                        if not filtered_df[col].isna().all():
                            filtered_df = filtered_df[
                                (filtered_df[col] >= temporal_filter['start_date']) &
                                (filtered_df[col] <= temporal_filter['end_date'])
                            ]
                            logger.info(f"Applied temporal filter on column {col}")
                            break
                    except Exception as e:
                        logger.debug(f"Could not convert column {col} to datetime: {str(e)}")
                        continue
            
            # Apply categorical filters
            categorical_filters = filter_state.get('categorical', {})
            for col, values in categorical_filters.items():
                if values:  # Only apply filter if values are selected
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
                    logger.info(f"Applied categorical filter on column {col}")
            
            # Create new figures with filtered data
            dashboard_builder = DashboardBuilder(filtered_df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            # Ensure we have the same number of figures as current_figures
            new_figures = []
            for i in range(len(current_figures)):
                if i < len(figures):
                    fig, _ = list(figures.values())[i]
                    new_figures.append(fig)
                else:
                    new_figures.append(current_figures[i])
            
            logger.info(f"Created {len(new_figures)} new figures with filtered data")
            
            # Return filtered state
            return (
                new_figures,
                temporal_filter.get('start_date'),
                temporal_filter.get('end_date'),
                [categorical_filters.get(cat_id['column'], []) for cat_id in category_ids]
            )
        
        # Return current state if no filters
        return current_figures, None, None, [[] for _ in category_ids]
        
    except Exception as e:
        logger.error(f"Error updating visualizations: {str(e)}")
        # Return current state on error
        return current_figures, None, None, [[] for _ in category_ids]

# Add callback to populate KPI selector when data is loaded
@app.callback(
    Output('kpi-selector', 'options'),
    Input('data-store', 'data'),
    prevent_initial_call=True
)
def update_kpi_selector(json_data):
    """
    Update KPI selector options based on loaded dataset columns.
    """
    if not json_data:
        return []
        
    try:
        data_store = json.loads(json_data)
        df = pd.read_json(io.StringIO(data_store['full_data']), orient='split')
        
        # Create options from column names
        options = [{'label': col, 'value': col} for col in df.columns]
        return options
        
    except Exception as e:
        logger.error(f"Error updating KPI selector: {str(e)}")
        return []

if __name__ == '__main__':
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )

