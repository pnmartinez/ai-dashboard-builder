import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from ai_dashboard_builder.llm.llm_pipeline import LLMPipeline


class TestLLMPipelineLocal(unittest.TestCase):
    """Tests for LLMPipeline operating in local (Ollama) mode."""

    def setUp(self):
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'sales': [100, 150, 120, 180, 160],
            'category': ['A', 'B', 'A', 'B', 'A']
        })

        self.pipeline = LLMPipeline(model_name="test-model", use_local=True)

    @patch('ai_dashboard_builder.llm.llm_pipeline.LLMPipeline._query_local')
    def test_analyze_dataset(self, mock_query):
        """Test dataset analysis delegates to _query_local."""
        mock_query.return_value = "Test analysis response"

        result = self.pipeline.analyze_dataset(self.df)
        self.assertIsInstance(result, str)
        mock_query.assert_called_once()

    @patch('ai_dashboard_builder.llm.llm_pipeline.LLMPipeline._query_local')
    def test_suggest_visualizations_returns_dict(self, mock_query):
        """Test visualization suggestion returns a dict (or falls back to empty)."""
        mock_query.return_value = '{"viz_1": {"type": "bar", "x": "category", "y": "sales", "title": "T", "description": "D", "parameters": {}}}'

        result = self.pipeline.suggest_visualizations(self.df)
        self.assertIsInstance(result, dict)

    # ------------------------------------------------------------------
    # Rate-limit delay helpers
    # ------------------------------------------------------------------

    def test_rate_limit_delay_openai_model(self):
        """GPT models use the openai rate limit."""
        p = LLMPipeline(model_name="gpt-4o", use_local=True)
        self.assertEqual(p._get_rate_limit_delay(), p.rate_limits["openai"])

    def test_rate_limit_delay_claude_model(self):
        """Claude models use the anthropic rate limit."""
        p = LLMPipeline(model_name="claude-3-sonnet", use_local=True)
        self.assertEqual(p._get_rate_limit_delay(), p.rate_limits["anthropic"])

    def test_rate_limit_delay_mistral_model(self):
        """Mistral models use the mistral rate limit."""
        p = LLMPipeline(model_name="mistral-large", use_local=True)
        self.assertEqual(p._get_rate_limit_delay(), p.rate_limits["mistral"])

    def test_rate_limit_delay_llama_model(self):
        """Llama models use the groq rate limit."""
        p = LLMPipeline(model_name="llama3.3-70b", use_local=True)
        self.assertEqual(p._get_rate_limit_delay(), p.rate_limits["groq"])

    def test_rate_limit_delay_unknown_model(self):
        """Unknown model names fall back to the default rate limit."""
        p = LLMPipeline(model_name="some-unknown-model", use_local=True)
        self.assertEqual(p._get_rate_limit_delay(), p.rate_limits["default"])

    # ------------------------------------------------------------------
    # _serialize_for_json
    # ------------------------------------------------------------------

    def test_serialize_numpy_int(self):
        """numpy int64 values are serialised to plain Python int."""
        result = self.pipeline._serialize_for_json(np.int64(42))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

    def test_serialize_numpy_float(self):
        """numpy float64 values are serialised to plain Python float."""
        result = self.pipeline._serialize_for_json(np.float64(3.14))
        self.assertIsInstance(result, float)

    def test_serialize_numpy_bool(self):
        """numpy bool_ values are serialised to plain Python bool."""
        result = self.pipeline._serialize_for_json(np.bool_(True))
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_serialize_pandas_timestamp(self):
        """Pandas Timestamp values are serialised to ISO-format strings."""
        ts = pd.Timestamp("2023-01-01")
        result = self.pipeline._serialize_for_json(ts)
        self.assertIsInstance(result, str)
        self.assertIn("2023-01-01", result)

    def test_serialize_nan_returns_none(self):
        """NaN values are converted to None."""
        result = self.pipeline._serialize_for_json(float("nan"))
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _query_local – HTTP-level behaviour
    # ------------------------------------------------------------------

    @patch('ai_dashboard_builder.llm.llm_pipeline.requests.post')
    def test_query_local_successful_response(self, mock_post):
        """_query_local returns the model response on success."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "hello from ollama"}
        mock_post.return_value = mock_response

        result = self.pipeline._query_local("test prompt")
        self.assertEqual(result, "hello from ollama")

    @patch('ai_dashboard_builder.llm.llm_pipeline.requests.post')
    def test_query_local_http_error(self, mock_post):
        """_query_local returns an error string on non-OK HTTP status."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = self.pipeline._query_local("test prompt")
        self.assertTrue(result.startswith("Error:"))

    @patch('ai_dashboard_builder.llm.llm_pipeline.requests.post')
    def test_query_local_unexpected_format_returns_error_string(self, mock_post):
        """_query_local returns an error string on unexpected response format.

        The ValueError raised internally is caught by the broad except-block
        so the method always returns a string rather than propagating the exception.
        """
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_key": "value"}
        mock_post.return_value = mock_response

        result = self.pipeline._query_local("test prompt")
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"))


class TestLLMPipelineExternal(unittest.TestCase):
    """Tests for LLMPipeline operating with an external API key."""

    def test_init_with_openai_key(self):
        """Initialising with an OpenAI model and key succeeds."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            p = LLMPipeline(model_name="gpt-4o", use_local=False)
            self.assertEqual(p.api_key, "sk-test123")

    def test_init_with_anthropic_key(self):
        """Initialising with a Claude model and key succeeds."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'ant-test'}):
            p = LLMPipeline(model_name="claude-3-opus", use_local=False)
            self.assertEqual(p.api_key, "ant-test")

    def test_init_missing_api_key_raises(self):
        """Missing API key raises ValueError."""
        with patch.dict('os.environ', {}, clear=True):
            # Ensure the key is not set
            import os
            os.environ.pop('OPENAI_API_KEY', None)
            with self.assertRaises(ValueError):
                LLMPipeline(model_name="gpt-4o", use_local=False)


if __name__ == '__main__':
    unittest.main()
