import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from ai_dashboard_builder.llm.llm_pipeline import LLMPipeline

class TestLLMPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'sales': [100, 150, 120, 180, 160],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        self.pipeline = LLMPipeline(model_name="test-model", use_local=True)

    @patch('ai_dashboard_builder.llm.llm_pipeline.LLMPipeline._query_local')
    def test_analyze_dataset(self, mock_query):
        """Test dataset analysis"""
        mock_query.return_value = "Test analysis response"
        
        result = self.pipeline.analyze_dataset(self.df)
        self.assertIsInstance(result, str)
        mock_query.assert_called_once()


if __name__ == '__main__':
    unittest.main()