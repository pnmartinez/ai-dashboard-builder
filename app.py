import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, no_update, long_callback, MATCH
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
    
    # Main container
    dbc.Container(fluid=True, children=[
        html.H1('AI Dashboard Builder', 
            style={
                'textAlign': 'center', 
                'color': COLORS['primary'], 
                'marginBottom': '2rem',
                'paddingTop': '1rem'
            }
        ),
        
        # Responsive Row
        dbc.Row([
            # Controls Column - Full width on mobile, 1/4 on desktop
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # LLM Provider Selection
                        html.Div([
                            html.H5("LLM Provider", className="mb-3"),
                            dbc.RadioItems(
                                id='llm-provider',
                                options=[
                                    {'label': html.A('Local (Ollama)', href='https://ollama.com/download', target='_blank'), 'value': 'local'},
                                    {'label': 'External API', 'value': 'external'}
                                ],
                                value='local',
                                className="mb-3"
                            ),
                            
                            # API Key Input (initially hidden)
                            dbc.Collapse(
                                dbc.Input(
                                    id='api-key-input',
                                    type='password',
                                    placeholder='Enter API Key',
                                    className="mb-3"
                                ),
                                id='api-key-collapse',
                                is_open=False
                            ),
                            
                            # Model Selection for External API
                            dbc.Collapse(
                                dbc.Select(
                                    id='model-selection',
                                    options=[
                                        {'label': 'GPT-4', 'value': 'gpt-4o-mini'},
                                        {'label': 'GPT-3.5', 'value': 'gpt-3.5-turbo'},
                                        {'label': 'Claude Sonnet 3.5', 'value': 'claude-3-sonnet-20240229'},
                                        {'label': 'Mistral Large', 'value': 'mistral-large'}
                                    ],
                                    value='gpt-4o-mini',
                                    className="mb-3"
                                ),
                                id='model-selection-collapse',
                                is_open=False
                            ),
                        ], className="mb-4"),
                        
                        # File Upload
                        html.Div([
                            html.H5("Dataset Upload", className="mb-3"),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select a CSV File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0',
                                    'backgroundColor': COLORS['background']
                                },
                                multiple=False
                            ),
                            
                            html.Div(id='upload-status', className="mt-2"),
                        ], className="mb-4"),
                        
                        # Analysis Button
                        dbc.Button(
                            'Analyze Data',
                            id='analyze-button',
                            color='primary',
                            className='w-100',
                            disabled=True
                        ),
                    ])
                ], className="mb-3")
            ], xs=12, md=4, className="mb-4"),  # Full width on mobile (<768px), 1/3 on desktop
            
            # Results Column - Full width on mobile, 9/12 on desktop
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
            ], xs=12, md=8, lg=9, className="px-2"),  # Full width on mobile (<768px), 9/12 on desktop
        ], className="g-0"),  # Remove gutters from row
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

# Modify the file upload callback to include the upload container visibility
@app.callback(
    [Output('data-store', 'data'),
     Output('upload-status', 'children'),
     Output('analyze-button', 'disabled'),
     Output('upload-data', 'style')],  # Add style output
    Input('upload-data', 'contents'),
    [State('upload-data', 'filename'),
     State('upload-data', 'style')],  # Add style state
    prevent_initial_call=True
)
def handle_upload(contents, filename, current_style):
    if contents is None:
        return no_update, no_update, True, current_style
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' not in filename.lower():
            return None, html.Div('Please upload a CSV file', style={'color': COLORS['error']}), True, current_style
            
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if df.empty:
            return None, html.Div('The uploaded file is empty', style={'color': COLORS['error']}), True, current_style
        
        # Hide the upload component after successful upload
        hidden_style = {**current_style, 'display': 'none'}
            
        return (
            df.to_json(date_format='iso', orient='split'),
            html.Div([
                html.Div(f'Loaded: {filename}', style={'color': COLORS['info']}),
                html.Button(
                    'Change File',
                    id='change-file-button',
                    className='mt-2 btn btn-outline-secondary btn-sm',
                    n_clicks=0
                )
            ]),
            False,
            hidden_style
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return None, html.Div(f'Error: {str(e)}', style={'color': COLORS['error']}), True, current_style

# Add callback for the change file button
@app.callback(
    [Output('upload-data', 'style', allow_duplicate=True),
     Output('data-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children', allow_duplicate=True),
     Output('analyze-button', 'disabled', allow_duplicate=True)],
    Input('change-file-button', 'n_clicks'),
    State('upload-data', 'style'),
    prevent_initial_call=True
)
def change_file(n_clicks, current_style):
    if n_clicks:
        # Show the upload component again
        visible_style = {**current_style, 'display': 'block'}
        return visible_style, None, '', True
    return no_update, no_update, no_update, no_update

# Modify the analyze callback to include provider and API key
@app.long_callback(
    Output('results-container', 'children'),
    Input('analyze-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('llm-provider', 'value'),
     State('api-key-input', 'value'),
     State('model-selection', 'value')],
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
def analyze_data(set_progress, n_clicks, json_data, provider, api_key, model):
    if not n_clicks or not json_data:
        raise PreventUpdate
    
    try:
        set_progress(html.Div("Loading data...", style={'color': COLORS['info']}))
        df = pd.read_json(io.StringIO(json_data), orient='split')
        
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
        
        # Rest of the analysis process remains the same
        set_progress(html.Div("1/5 Analyzing dataset...", style={'color': COLORS['info']}))
        analysis = pipeline.analyze_dataset(df)
        
        set_progress(html.Div("2/5 Generating visualization suggestions...", style={'color': COLORS['info']}))
        viz_specs = pipeline.suggest_visualizations(df)
        
        set_progress(html.Div("3/5 Creating visualizations...", style={'color': COLORS['info']}))
        dashboard_builder = DashboardBuilder(df, COLORS)
        figures = dashboard_builder.create_all_figures(viz_specs)
        
        set_progress(html.Div("4/5 Generating insights summary...", style={'color': COLORS['info']}))
        summary = pipeline.summarize_analysis(analysis, viz_specs)
        
        set_progress(html.Div("5/5 Rendering dashboard...", style={'color': COLORS['info']}))
        return html.Div([
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
            
            # Key Insights Section (Second)
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
                            'fontFamily': 'inherit'  # Use regular font for better readability
                        }
                    )
                )
            ], className='mb-4'),
            
            # Dataset Analysis Section (Last)
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

if __name__ == '__main__':
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )
