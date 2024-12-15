import unittest
import pandas as pd
from ai_dashboard_builder.llm.prompts import create_dataset_analysis_prompt, create_visualization_prompt

class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3),
            'sales': [100, 150, 120],
            'category': ['A', 'B', 'A']
        })
        
        self.data_summary = {
            'columns': ['date', 'sales', 'category'],
            'sample_rows': self.df.head().to_dict('records'),
            'data_types': {'date': 'datetime64[ns]', 'sales': 'int64', 'category': 'object'},
            'null_counts': {'date': 0, 'sales': 0, 'category': 0},
            'unique_counts': {'date': 3, 'sales': 3, 'category': 2}
        }

    def test_dataset_analysis_prompt(self):
        """Test dataset analysis prompt generation"""
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary)
        
        self.assertIsInstance(prompt, str)
        self.assertIn("KEY OBSERVATIONS", prompt)
        self.assertIn("STATISTICAL HIGHLIGHTS", prompt)
        self.assertIn("RECOMMENDATIONS", prompt)

    def test_dataset_analysis_prompt_with_kpis(self):
        """Test dataset analysis prompt with KPIs"""
        kpis = ['sales']
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary, kpis)
        
        self.assertIn("Relationship to KPIs", prompt)
        self.assertIn("KPI drivers", prompt)

if __name__ == '__main__':
    unittest.main()