import unittest
import pandas as pd
import plotly.graph_objects as go
from ai_dashboard_builder.dashboard_builder import DashboardBuilder, code_block_to_lines


class TestDashboardBuilder(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'sales': [100, 150, 120, 180, 160],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'quantity': [10, 15, 12, 18, 16],
            'begin': pd.date_range(start='2023-01-01', periods=5),
            'end': pd.date_range(start='2023-01-02', periods=5),
        })

        self.colors = {
            "background": "#FFF5F5",
            "primary": "#FF9999",
            "secondary": "#FF7777"
        }

        self.builder = DashboardBuilder(self.df, self.colors)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization(self):
        """Test DashboardBuilder initialization."""
        self.assertIsInstance(self.builder.df, pd.DataFrame)
        self.assertEqual(self.builder.colors, self.colors)

    # ------------------------------------------------------------------
    # Line plot
    # ------------------------------------------------------------------

    def test_create_line_plot(self):
        """Test line plot creation."""
        viz_spec = {
            "type": "line",
            "x": "date",
            "y": "sales",
            "title": "Sales Over Time",
            "parameters": {"height": 400}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIsInstance(code, str)
        self.assertIn("import plotly.express as px", code)

    # ------------------------------------------------------------------
    # Bar plot
    # ------------------------------------------------------------------

    def test_create_bar_plot(self):
        """Test bar plot creation."""
        viz_spec = {
            "type": "bar",
            "x": "category",
            "y": "sales",
            "title": "Sales by Category",
            "parameters": {"barmode": "group"}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.bar", code)

    def test_create_bar_plot_with_count_aggregation(self):
        """Test bar plot with count aggregation falls back to histogram."""
        viz_spec = {
            "type": "bar",
            "x": "category",
            "y": "sales",
            "title": "Count by Category",
            "parameters": {"aggregation": "count"}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.histogram", code)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def test_create_histogram(self):
        """Test histogram creation."""
        viz_spec = {
            "type": "histogram",
            "x": "sales",
            "title": "Sales Distribution",
            "parameters": {"nbins": 10}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.histogram", code)

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------

    def test_create_scatter_plot(self):
        """Test scatter plot creation."""
        viz_spec = {
            "type": "scatter",
            "x": "sales",
            "y": "quantity",
            "title": "Sales vs Quantity",
            "parameters": {}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.scatter", code)

    def test_create_scatter_plot_with_numeric_size(self):
        """Test scatter plot with a numeric size parameter."""
        viz_spec = {
            "type": "scatter",
            "x": "sales",
            "y": "quantity",
            "title": "Sized Scatter",
            "parameters": {"size": 10}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)

    def test_create_scatter_plot_with_invalid_size_column(self):
        """Test scatter plot drops unknown size column gracefully."""
        viz_spec = {
            "type": "scatter",
            "x": "sales",
            "y": "quantity",
            "title": "Scatter Invalid Size",
            "parameters": {"size": "nonexistent_column"}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def test_create_heatmap(self):
        """Test heatmap creation."""
        viz_spec = {
            "type": "heatmap",
            "x": "category",
            "y": "category",
            "title": "Category Heatmap",
            "parameters": {}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.imshow", code)

    def test_create_heatmap_missing_axes_raises(self):
        """Test heatmap raises ValueError when x or y is missing."""
        viz_spec = {
            "type": "heatmap",
            "title": "Broken Heatmap",
            "parameters": {}
        }

        with self.assertRaises(ValueError):
            self.builder.create_figure(viz_spec)

    # ------------------------------------------------------------------
    # Box plot
    # ------------------------------------------------------------------

    def test_create_box_plot(self):
        """Test box plot creation."""
        viz_spec = {
            "type": "box",
            "x": "category",
            "y": "sales",
            "title": "Sales Distribution Box",
            "parameters": {}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.box", code)

    # ------------------------------------------------------------------
    # Violin plot
    # ------------------------------------------------------------------

    def test_create_violin_plot(self):
        """Test violin plot creation."""
        viz_spec = {
            "type": "violin",
            "x": "category",
            "y": "sales",
            "title": "Sales Violin",
            "parameters": {}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.violin", code)

    # ------------------------------------------------------------------
    # Pie chart
    # ------------------------------------------------------------------

    def test_create_pie_chart(self):
        """Test pie chart creation."""
        viz_spec = {
            "type": "pie",
            "x": "category",
            "y": "sales",
            "title": "Sales Share",
            "parameters": {"hole": 0.3}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.pie", code)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def test_create_timeline(self):
        """Test timeline creation when begin/end columns exist."""
        viz_spec = {
            "type": "timeline",
            "y": "category",
            "title": "Project Timeline",
            "parameters": {}
        }

        figure, code = self.builder.create_figure(viz_spec)
        self.assertIsInstance(figure, go.Figure)
        self.assertIn("px.timeline", code)

    def test_create_timeline_missing_columns_raises(self):
        """Test timeline raises ValueError without begin/end columns."""
        df_no_timeline = self.df.drop(columns=["begin", "end"])
        builder = DashboardBuilder(df_no_timeline, self.colors)

        viz_spec = {
            "type": "timeline",
            "y": "category",
            "title": "Broken Timeline",
            "parameters": {}
        }

        with self.assertRaises(ValueError):
            builder.create_figure(viz_spec)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_invalid_x_column(self):
        """Test handling of invalid x column name."""
        viz_spec = {
            "type": "line",
            "x": "nonexistent_column",
            "y": "sales",
            "title": "Invalid Plot"
        }

        with self.assertRaises(ValueError):
            self.builder.create_figure(viz_spec)

    def test_invalid_y_column(self):
        """Test handling of invalid y column name."""
        viz_spec = {
            "type": "line",
            "x": "date",
            "y": "nonexistent_column",
            "title": "Invalid Y"
        }

        with self.assertRaises(ValueError):
            self.builder.create_figure(viz_spec)

    def test_unsupported_viz_type_raises(self):
        """Test that an unsupported visualization type raises ValueError."""
        viz_spec = {
            "type": "radar",
            "x": "category",
            "y": "sales",
            "title": "Unsupported"
        }

        with self.assertRaises(ValueError):
            self.builder.create_figure(viz_spec)

    # ------------------------------------------------------------------
    # Batch creation
    # ------------------------------------------------------------------

    def test_create_all_figures(self):
        """Test batch figure creation, partial failures are skipped."""
        viz_specs = {
            "viz_1": {
                "type": "line",
                "x": "date",
                "y": "sales",
                "title": "Line"
            },
            "viz_2": {
                "type": "bar",
                "x": "category",
                "y": "sales",
                "title": "Bar"
            },
            "viz_bad": {
                "type": "line",
                "x": "nonexistent",
                "y": "sales",
                "title": "Should Fail"
            },
        }

        figures = self.builder.create_all_figures(viz_specs)
        # Two valid specs produce figures; the broken one is skipped
        self.assertIn("viz_1", figures)
        self.assertIn("viz_2", figures)
        self.assertNotIn("viz_bad", figures)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def test_code_block_formatting(self):
        """Test code block formatting utility."""
        code = """
            def test_function():
                print("hello")
                return True
        """
        formatted = code_block_to_lines(code)
        self.assertIsInstance(formatted, list)
        self.assertTrue(all(isinstance(line, str) for line in formatted))

    def test_code_block_strips_leading_trailing_blank_lines(self):
        """Blank lines at head/tail of code block are stripped."""
        code = "\n\n    x = 1\n\n"
        result = code_block_to_lines(code)
        self.assertEqual(result[0], "x = 1")
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
