from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging

logger = logging.getLogger('DashboardBuilder')

class DashboardBuilder:
    def __init__(self, df: pd.DataFrame, color_theme: Dict[str, str]):
        """
        Initialize the dashboard builder.
        
        Args:
            df (pd.DataFrame): The dataset to visualize
            color_theme (Dict[str, str]): Color theme dictionary
        """
        self.df = df
        self.colors = color_theme
        
    def create_figure(self, viz_spec: Dict[str, Any]) -> tuple[go.Figure, str]:
        """
        Convert a visualization specification into a Plotly figure and its code.
        
        Returns:
            tuple[go.Figure, str]: Plotly figure object and its Python code
        """
        try:
            viz_type = viz_spec.get('type', '').lower()
            
            # Get basic parameters
            x = viz_spec.get('x')
            y = viz_spec.get('y')
            color = viz_spec.get('color')
            title = viz_spec.get('title', 'Visualization')
            params = viz_spec.get('parameters', {})
            
            # Validate column names and adjust parameters if needed
            if x and x not in self.df.columns:
                logger.warning(f"Column '{x}' not found in dataframe, skipping visualization")
                raise ValueError(f"Column '{x}' not found in dataframe")
            
            if y and y not in self.df.columns:
                logger.warning(f"Column '{y}' not found in dataframe, skipping visualization")
                raise ValueError(f"Column '{y}' not found in dataframe")
            
            # Handle size parameter for scatter plots
            if viz_type == 'scatter' and 'size' in params:
                size_value = params['size']
                if isinstance(size_value, (int, float)):
                    # If size is a number, use it directly
                    size = size_value
                elif isinstance(size_value, str) and size_value not in self.df.columns:
                    # If size is a string but not a column name, remove it
                    logger.warning(f"Size column '{size_value}' not found, using default size")
                    size = None
                else:
                    size = size_value
            else:
                size = None
            
            # Create figure based on visualization type
            if viz_type == 'line':
                fig = px.line(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    markers=params.get('markers', True),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'bar':
                if params.get('aggregation') == 'count':
                    fig = px.histogram(
                        self.df,
                        x=x,
                        color=color if color in self.df.columns else None,
                        title=title,
                        barmode=params.get('barmode', 'relative'),
                        color_discrete_sequence=[color] if color and color not in self.df.columns else None
                    )
                else:
                    fig = px.bar(
                        self.df,
                        x=x,
                        y=y,
                        color=color if color in self.df.columns else None,
                        title=title,
                        barmode=params.get('barmode', 'relative'),
                        color_discrete_sequence=[color] if color and color not in self.df.columns else None
                    )
                    
            elif viz_type == 'histogram':
                fig = px.histogram(
                    self.df,
                    x=x,
                    color=color if color in self.df.columns else None,
                    title=title,
                    nbins=params.get('nbins', 30),
                    histnorm=params.get('histnorm', None),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'scatter':
                fig = px.scatter(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    size=size,
                    hover_data=params.get('hover_data', None),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'heatmap':
                # Prepare data for heatmap
                if x and y:
                    pivot_table = pd.crosstab(self.df[x], self.df[y])
                    fig = px.imshow(
                        pivot_table,
                        title=title,
                        color_continuous_scale=params.get('color_scale', 'RdBu_r'),
                        aspect=params.get('aspect', 'auto')
                    )
                else:
                    raise ValueError("Both x and y are required for heatmap")
                
            elif viz_type == 'box':
                fig = px.box(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    points=params.get('points', 'outliers'),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'violin':
                fig = px.violin(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    box=params.get('box', True),
                    points=params.get('points', 'outliers'),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'pie':
                fig = px.pie(
                    self.df,
                    names=x,
                    values=y,
                    title=title,
                    hole=params.get('hole', 0),
                    color_discrete_sequence=[color] if color and color not in self.df.columns else None
                )
                
            elif viz_type == 'timeline':
                if all(col in self.df.columns for col in ['begin', 'end']):
                    fig = px.timeline(
                        self.df,
                        x_start='begin',
                        x_end='end',
                        y=y or 'name',
                        color=color if color in self.df.columns else None,
                        title=title,
                        color_discrete_sequence=[color] if color and color not in self.df.columns else None
                    )
                else:
                    raise ValueError("Timeline requires 'begin' and 'end' columns")
                
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Generate Python code for the visualization
            code_lines = []
            code_lines.append("import plotly.express as px")
            code_lines.append("import plotly.graph_objects as go")
            code_lines.append("")
            
            if viz_type == 'line':
                code_lines.append(f"""fig = px.line(
    df,
    x="{x}",
    y="{y}",
    color="{color}" if "{color}" else None,
    title="{title}",
    markers={params.get('markers', True)}
)""")
            # ... (add similar code generation for other chart types) ...
            
            # Add styling code
            code_lines.append("""
# Apply styling
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font={'color': "#4A4A4A"},
    title_font_color="#4A4A4A",
    showlegend=True,
    legend=dict(
        bgcolor="white",
        bordercolor="#FFD7D7",
        borderwidth=1
    ),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Update axes
fig.update_xaxes(
    gridcolor="#FFD7D7",
    showgrid=True,
    zeroline=False
)
fig.update_yaxes(
    gridcolor="#FFD7D7",
    showgrid=True,
    zeroline=False
)""")
            
            code = "\n".join(code_lines)
            return fig, code
            
        except Exception as e:
            logger.error(f"Error creating {viz_type} visualization: {str(e)}")
            raise

    def create_all_figures(self, viz_specs: Dict[str, Any]) -> Dict[str, tuple[go.Figure, str]]:
        """
        Convert all visualization specifications into Plotly figures and their code.
        
        Returns:
            Dict[str, tuple[go.Figure, str]]: Dictionary of Plotly figures and their code
        """
        figures = {}
        for viz_id, viz_spec in viz_specs.items():
            try:
                fig, code = self.create_figure(viz_spec)
                figures[viz_id] = (fig, code)
                logger.info(f"Successfully created figure for {viz_id}")
            except Exception as e:
                logger.error(f"Error creating figure for {viz_id}: {str(e)}")
                continue
        return figures 
