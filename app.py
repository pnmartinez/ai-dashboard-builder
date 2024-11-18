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
    
    # Main container
    dbc.Container(fluid=True, children=[
        html.A(
            html.H1('AI Dashboard Builder',
                style={
                    'textAlign': 'center',
                    'color': COLORS['primary'],
                    'marginBottom': '2rem',
                    'paddingTop': '1rem',
                    'textDecoration': 'none'  # Remove underline from link
                }
            ),
            href='/',  # Link to homepage
            style={'textDecoration': 'none'}  # Remove underline from link container
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
                                 #   {'label': 'Claude Sonnet 3.5', 'value': 'claude-3-sonnet-20240229'},
                                  #  {'label': 'Mistral Large', 'value': 'mistral-large'}
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
                        dbc.Checkbox(
                            id='viz-only-checkbox',
                            label="Visualizations only (faster process)",
                            value=False,
                            className="mt-2",
                            style={'color': '#6c757d'}  # Using a muted secondary text color
                        ),
                    ], xs=12, md=3, className="d-flex align-items-end flex-column"),
                ])
            ])
        ], className="mb-4"),
        
        # Results Section - Full width
        dbc.Spinner(
            html.Div(
                id='results-container',
                style={'minHeight': '200px'}
            ),
            color='primary',
            type='border',
            fullscreen=False,
        )
    ])
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# Callback to toggle API key and model selection visibility
@app.callback(
    [Output('api-key-collapse', 'is_open'),
     Output('model-selection-collapse', 'is_open')],
    Input('llm-provider', 'value')
)
def toggle_api_key(provider):
    show_api = provider == 'external'
    return show_api, show_api

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
        
        # Store the full dataframe and initial limits - now using max values
        data_store = {
            'full_data': df.to_json(date_format='iso', orient='split'),
            'row_limit': len(df),  # Use max rows by default
            'col_limit': len(df.columns)  # Use max columns by default
        }
        
        # Hide the upload component after successful upload
        hidden_style = {**current_style, 'display': 'none'}

        # Create preview controls with max values by default
        preview_controls = dbc.Row([
            dbc.Col([
                html.Label("Limit Rows:", className="me-2"),
                dbc.Input(
                    id='preview-rows-input',
                    type='number',
                    min=1,
                    max=len(df),
                    value=len(df),  # Set to max by default
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
                    value=len(df.columns),  # Set to max by default
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

        # Create initial preview table with max limits
        preview_df = df  # Use full dataframe for preview
        preview_table = dbc.Table.from_dataframe(
            preview_df,
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            style={'backgroundColor': 'white'}
        )
            
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
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'import-viz-specs-button':
        # Use absolute path for viz specs directory
        viz_specs_dir = os.path.join(BASE_DIR, 'llm_responses')
        viz_specs_files = glob.glob(os.path.join(viz_specs_dir, 'viz_specs_*.json'))
        viz_specs_files.sort(reverse=True)  # Most recent first
        
        if not viz_specs_files:
            return True, html.Div("No visualization specifications found", className="text-muted")
        
        # Create list of files with metadata
        file_list = []
        for file_path in viz_specs_files:
            try:
                with open(file_path, 'r') as f:
                    specs = json.load(f)
                    timestamp = specs.get('timestamp', 'Unknown')
                    model = specs.get('model', 'Unknown')
                    provider = specs.get('provider', 'Unknown')
                    
                    file_list.append(
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.H6(f"Generated: {timestamp}", className="mb-1"),
                                        html.Small(f"Model: {model} ({provider})", className="text-muted")
                                    ]
                                ),
                                dbc.Button(
                                    "Use",
                                    id={'type': 'use-viz-specs', 'index': file_path},
                                    color="primary",
                                    size="sm",
                                    className="ms-auto"
                                )
                            ],
                            className="d-flex justify-content-between align-items-center"
                        )
                    )
            except Exception as e:
                logger.error(f"Error reading viz specs file {file_path}: {str(e)}")
                continue
        
        return True, dbc.ListGroup(file_list)
    
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

# Modify the analyze_data callback to properly handle imported specs
@app.long_callback(
    Output('results-container', 'children'),
    Input('analyze-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('llm-provider', 'value'),
     State('api-key-input', 'value'),
     State('model-selection', 'value'),
     State('viz-only-checkbox', 'value')],
    prevent_initial_call=True,
    running=[
        (Output('analyze-button', 'disabled'), True, False),
        (Output('upload-data', 'disabled'), True, False),
        (Output('llm-provider', 'disabled'), True, False),
        (Output('api-key-input', 'disabled'), True, False),
        (Output('model-selection', 'disabled'), True, False),
    ],
    progress=[Output('upload-status', 'children')],
)
def analyze_data(set_progress, n_clicks, json_data, provider, api_key, model, viz_only):
    if not n_clicks or not json_data:
        raise PreventUpdate
    
    try:
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
                set_progress(html.Div("1/5 Analyzing dataset...", style={'color': COLORS['info']}))
                analysis = pipeline.analyze_dataset(df)
            else:
                analysis = "Analysis skipped - visualizations only mode"
            
            set_progress(html.Div("2/5 Generating visualization suggestions...", style={'color': COLORS['info']}))
            viz_specs = pipeline.suggest_visualizations(df)
            
            set_progress(html.Div("3/5 Creating visualizations...", style={'color': COLORS['info']}))
            dashboard_builder = DashboardBuilder(df, COLORS)
            figures = dashboard_builder.create_all_figures(viz_specs)
            
            # Skip summary if viz_only is True
            if not viz_only:
                set_progress(html.Div("4/5 Generating insights summary...", style={'color': COLORS['info']}))
                summary = pipeline.summarize_analysis(analysis, viz_specs)
            else:
                summary = "Summary skipped - visualizations only mode"
        
        set_progress(html.Div("5/5 Rendering dashboard...", style={'color': COLORS['info']}))
        
        components = [
            # Visualizations Section (First)
            dbc.Card([
                dbc.CardHeader(
                    html.H3("Visualizations", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            # Tabs for each visualization
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
                                    # Content for both tabs
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
                        ], width=12) for i, (fig, code) in enumerate(figures.values())
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
                        html.H3("Key Insights", className="mb-0")
                    ),
                    dbc.CardBody(
                        html.Pre(
                            summary,
                            style={
                                'whiteSpace': 'pre-wrap',
                                'backgroundColor': COLORS['background'],
                                'padding': '1rem',
                                'borderRadius': '5px',
                                'fontFamily': 'inherit'
                            }
                        )
                    )
                ], className='mb-4'),
                
                # Dataset Analysis Section
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Dataset Analysis", className="mb-0")
                    ),
                    dbc.CardBody(
                        html.Pre(
                            analysis,
                            style={
                                'whiteSpace': 'pre-wrap',
                                'backgroundColor': COLORS['background'],
                                'padding': '1rem',
                                'borderRadius': '5px'
                            }
                        )
                    )
                ])
            ])
        
        return html.Div(components)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return html.Div(
            f"Error during analysis: {str(e)}", 
            style={'color': COLORS['error'], 'padding': '1rem'}
        )

# Fix the tab switching callback
@app.callback(
    [Output({'type': 'chart-content', 'index': MATCH}, 'style'),
     Output({'type': 'code-content', 'index': MATCH}, 'style')],
    Input({'type': 'tabs', 'index': MATCH}, 'active_tab'),
    prevent_initial_call=True
)
def switch_tab(active_tab):
    if active_tab and 'chart-tab' in active_tab:
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

# Add callback for code copying
app.clientside_callback(
    """
    function(n_clicks, code_content) {
        if (n_clicks) {
            navigator.clipboard.writeText(code_content.props.children[0].props.children);
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
def toggle_preview_visibility(viz_active, change_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'viz-state' and viz_active:
        return {'display': 'none'}
    elif trigger_id == 'change-file-button':
        return {'display': 'block'}
    
    return dash.no_update

if __name__ == '__main__':
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )
