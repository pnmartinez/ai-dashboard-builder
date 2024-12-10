"""Create interactive dashboards with Plotly visualizations from specifications."""

import logging
import textwrap
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("DashboardBuilder")


def code_block_to_lines(code: str) -> str:
    """Format a code block with consistent indentation and whitespace.

    Args:
        code (str): The code to format

    Returns:
        list[str]: The formatted code as a list of lines
    """
    return textwrap.dedent(code).strip().splitlines()


class DashboardBuilder:
    """A class for creating interactive dashboards with Plotly visualizations.

    This class handles the creation of various types of plots (line, bar, scatter, etc.)
    from visualization specifications, including both the figures and their reproducible code.

    Attributes:
        df (pd.DataFrame): The dataset to visualize
        colors (Dict[str, str]): Color theme dictionary for consistent styling
    """

    def __init__(self, df: pd.DataFrame, color_theme: Dict[str, str]):
        """Initialize the dashboard builder.

        Args:
            df (pd.DataFrame): The dataset to visualize
            color_theme (Dict[str, str]): Color theme dictionary for styling
        """
        self.df = df
        self.colors = color_theme

    def create_figure(self, viz_spec: Dict[str, Any]) -> tuple[go.Figure, str]:
        """Convert a visualization specification into a Plotly figure and its reproducible code.

        Args:
            viz_spec (Dict[str, Any]): Specification for the visualization including:
                - type: The type of visualization (line, bar, scatter, etc.)
                - x: Column name for x-axis
                - y: Column name for y-axis (if applicable)
                - color: Column name for color encoding or specific color
                - title: Title for the visualization
                - parameters: Additional parameters for the specific plot type

        Returns:
            tuple[go.Figure, str]: A tuple containing:
                - The Plotly figure object
                - Python code that reproduces the visualization

        Raises:
            ValueError: If required columns are not found or visualization type is unsupported
        """
        try:
            viz_type = viz_spec.get("type", "").lower()

            # Get basic parameters
            x = viz_spec.get("x")
            y = viz_spec.get("y")
            color = viz_spec.get("color")
            title = viz_spec.get("title", "Visualization")
            params = viz_spec.get("parameters", {})

            # Validate column names and adjust parameters if needed
            if x and x not in self.df.columns:
                logger.warning(
                    f"Column '{x}' not found in dataframe, skipping visualization"
                )
                raise ValueError(f"Column '{x}' not found in dataframe")

            if y and y not in self.df.columns:
                logger.warning(
                    f"Column '{y}' not found in dataframe, skipping visualization"
                )
                raise ValueError(f"Column '{y}' not found in dataframe")

            # Handle size parameter for scatter plots
            if viz_type == "scatter" and "size" in params:
                size_value = params["size"]
                if isinstance(size_value, (int, float)):
                    # If size is a number, use it directly
                    size = size_value
                elif isinstance(size_value, str) and size_value not in self.df.columns:
                    # If size is a string but not a column name, remove it
                    logger.warning(
                        f"Size column '{size_value}' not found, using default size"
                    )
                    size = None
                else:
                    size = size_value
            else:
                size = None

            # Create figure based on visualization type and generate corresponding code
            code_lines = [
                "import plotly.express as px",
                "import plotly.graph_objects as go",
                "import pandas as pd",
                "",
                "# Sample data preparation",
                "df = pd.DataFrame({",
                f"    '{x}': {list(self.df[x].head()) if x else []},",
            ]

            if y:
                code_lines.append(f"    '{y}': {list(self.df[y].head())},")
            if color and color in self.df.columns:
                code_lines.append(f"    '{color}': {list(self.df[color].head())},")

            code_lines.append("})")
            code_lines.append("")

            if viz_type == "line":
                fig = px.line(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    markers=params.get("markers", True),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )
                lines = f"""
                # Create line plot
                fig = px.line(
                    df,
                    x='{x}',
                    y='{y}',
                    color='{color}' if '{color}' in df.columns else None,
                    title='{title}',
                    markers={params.get('markers', True)}
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "bar":
                if params.get("aggregation") == "count":
                    fig = px.histogram(
                        self.df,
                        x=x,
                        color=color if color in self.df.columns else None,
                        title=title,
                        barmode=params.get("barmode", "relative"),
                    )

                    lines = f"""
                    # Create histogram
                    fig = px.histogram(
                        df,
                        x='{x}',
                        color='{color}' if '{color}' in df.columns else None,
                        title='{title}',
                        barmode='{params.get('barmode', 'relative')}'
                    )
                    """
                    code_lines.extend(code_block_to_lines(lines))
                else:
                    fig = px.bar(
                        self.df,
                        x=x,
                        y=y,
                        color=color if color in self.df.columns else None,
                        title=title,
                        barmode=params.get("barmode", "relative"),
                    )
                    lines = f"""
                    # Create bar plot
                    fig = px.bar(
                        df,
                        x='{x}',
                        y='{y}',
                        color='{color}' if '{color}' in df.columns else None,
                        title='{title}',
                        barmode='{params.get('barmode', 'relative')}'
                    )
                    """
                    code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "histogram":
                fig = px.histogram(
                    self.df,
                    x=x,
                    color=color if color in self.df.columns else None,
                    title=title,
                    nbins=params.get("nbins", 30),
                    histnorm=params.get("histnorm", None),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )

                lines = f"""
                # Create histogram
                fig = px.histogram(
                    df,
                    x='{x}',
                    color='{color}' if '{color}' in df.columns else None,
                    title='{title}',
                    nbins={params.get('nbins', 30)},
                    histnorm='{params.get('histnorm', None)}' if '{params.get('histnorm')}' else None
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "scatter":
                fig = px.scatter(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    size=size,
                    hover_data=params.get("hover_data", None),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )

                lines = f"""
                # Create scatter plot
                fig = px.scatter(
                    df,
                    x='{x}',
                    y='{y}',
                    color='{color}' if '{color}' in df.columns else None,
                    title='{title}',
                    size={size if isinstance(size, (int, float)) else f'{size}' if size else 'None'},
                    hover_data={params.get('hover_data')}
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "heatmap":
                # Prepare data for heatmap
                if x and y:
                    pivot_table = pd.crosstab(self.df[x], self.df[y])
                    fig = px.imshow(
                        pivot_table,
                        title=title,
                        color_continuous_scale=params.get("color_scale", "RdBu_r"),
                        aspect=params.get("aspect", "auto"),
                    )

                    lines = f"""
                    # Create heatmap
                    pivot_table = pd.crosstab(df['{x}'], df['{y}'])
                    fig = px.imshow(
                        pivot_table,
                        title='{title}',
                        color_continuous_scale='{params.get('color_scale', 'RdBu_r')}',
                        aspect='{params.get('aspect', 'auto')}'
                    )
                    """
                    code_lines.extend(code_block_to_lines(lines))
                else:
                    raise ValueError("Both x and y are required for heatmap")

            elif viz_type == "box":
                fig = px.box(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    points=params.get("points", "outliers"),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )

                lines = f"""
                # Create box plot
                fig = px.box(
                    df,
                    x='{x}',
                    y='{y}',
                    color='{color}' if '{color}' in df.columns else None,
                    title='{title}',
                    points='{params.get('points', 'outliers')}'
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "violin":
                fig = px.violin(
                    self.df,
                    x=x,
                    y=y,
                    color=color if color in self.df.columns else None,
                    title=title,
                    box=params.get("box", True),
                    points=params.get("points", "outliers"),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )

                lines = f"""
                # Create violin plot
                fig = px.violin(
                    df,
                    x='{x}',
                    y='{y}',
                    color='{color}' if '{color}' in df.columns else None,
                    title='{title}',
                    box={params.get('box', True)},
                    points='{params.get('points', 'outliers')}'
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "pie":
                fig = px.pie(
                    self.df,
                    names=x,
                    values=y,
                    title=title,
                    hole=params.get("hole", 0),
                    color_discrete_sequence=[color]
                    if color and color not in self.df.columns
                    else None,
                )
                lines = f"""
                # Create pie chart
                fig = px.pie(
                    df,
                    names='{x}',
                    values='{y}',
                    title='{title}',
                    hole={params.get('hole', 0)}
                )
                """
                code_lines.extend(code_block_to_lines(lines))

            elif viz_type == "timeline":
                if all(col in self.df.columns for col in ["begin", "end"]):
                    fig = px.timeline(
                        self.df,
                        x_start="begin",
                        x_end="end",
                        y=y or "name",
                        color=color if color in self.df.columns else None,
                        title=title,
                        color_discrete_sequence=[color]
                        if color and color not in self.df.columns
                        else None,
                    )
                    lines = f"""
                    # Create timeline
                    fig = px.timeline(
                        df,
                        x_start='begin',
                        x_end='end',
                        y='{y or 'name'}',
                        color='{color}' if '{color}' in df.columns else None,
                        title='{title}'
                    )
                    """
                    code_lines.extend(code_block_to_lines(lines))
                else:
                    raise ValueError("Timeline requires 'begin' and 'end' columns")

            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")

            # Add styling code
            lines = """
            # Apply styling
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': '#4A4A4A'},
                title_font_color='#4A4A4A',
                showlegend=True,
                legend=dict(
                    bgcolor='white',
                    bordercolor='#FFD7D7',
                    borderwidth=1
                ),
                margin=dict(l=10, r=10, t=30, b=10)
            )

            # Update axes
            fig.update_xaxes(gridcolor='#FFD7D7', showgrid=True, zeroline=False)
            fig.update_yaxes(gridcolor='#FFD7D7', showgrid=True, zeroline=False)

            # Show the plot
            fig.show()
            """
            code_lines.extend(code_block_to_lines(lines))

            code = "\n".join(code_lines)
            return fig, code

        except Exception as e:
            logger.error(f"Error creating {viz_type} visualization: {str(e)}")
            raise

    def create_all_figures(
        self, viz_specs: Dict[str, Any]
    ) -> Dict[str, tuple[go.Figure, str]]:
        """Convert all visualization specifications into Plotly figures and their code.

        Args:
            viz_specs (Dict[str, Any]): Dictionary of visualization specifications,
                where each value is a specification as described in create_figure()

        Returns:
            Dict[str, tuple[go.Figure, str]]: Dictionary mapping visualization IDs to tuples of:
                - Plotly figure object
                - Python code to reproduce the visualization

        Note:
            If creation of any individual visualization fails, it will be logged and skipped,
            allowing the rest of the visualizations to be created.
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
