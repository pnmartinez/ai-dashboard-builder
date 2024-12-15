import unittest
import pandas as pd
import plotly.graph_objects as go
from ai_dashboard_builder.dashboard_builder import DashboardBuilder, code_block_to_lines

class TestDashboardBuilder(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'sales': [100, 150, 120, 180, 160],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'quantity': [10, 15, 12, 18, 16]
        })
        
        self.colors = {
            "background": "#FFF5F5",
            "primary": "#FF9999",
            "secondary": "#FF7777"
        }
        
        self.builder = DashboardBuilder(self.df, self.colors)

    def test_initialization(self):
        """Test DashboardBuilder initialization"""
        self.assertIsInstance(self.builder.df, pd.DataFrame)
        self.assertEqual(self.builder.colors, self.colors)

    def test_create_line_plot(self):
        """Test line plot creation"""
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

    def test_create_bar_plot(self):
        """Test bar plot creation"""
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

    def test_invalid_column(self):
        """Test handling of invalid column names"""
        viz_spec = {
            "type": "line",
            "x": "nonexistent_column",
            "y": "sales",
            "title": "Invalid Plot"
        }
        
        with self.assertRaises(ValueError):
            self.builder.create_figure(viz_spec)

    def test_code_block_formatting(self):
        """Test code block formatting utility"""
        code = """
            def test_function():
                print("hello")
                return True
        """
        formatted = code_block_to_lines(code)
        self.assertIsInstance(formatted, list)
        self.assertTrue(all(isinstance(line, str) for line in formatted))

if __name__ == '__main__':
    unittest.main()