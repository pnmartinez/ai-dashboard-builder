import unittest
import pandas as pd
from ai_dashboard_builder.llm.prompts import (
    create_dataset_analysis_prompt,
    create_visualization_prompt,
    create_pattern_explanation_prompt,
    create_analysis_summary_prompt,
)


class TestDatasetAnalysisPrompt(unittest.TestCase):
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

    def test_returns_string(self):
        """Prompt is a non-empty string."""
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_contains_required_sections(self):
        """Prompt contains the mandatory structural sections."""
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary)

        self.assertIn("KEY OBSERVATIONS", prompt)
        self.assertIn("STATISTICAL HIGHLIGHTS", prompt)
        self.assertIn("RECOMMENDATIONS", prompt)

    def test_with_kpis_adds_kpi_context(self):
        """Prompt with KPIs includes KPI-specific sections."""
        kpis = ['sales']
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary, kpis)

        self.assertIn("Relationship to KPIs", prompt)
        self.assertIn("KPI drivers", prompt)

    def test_without_kpis_omits_kpi_context(self):
        """Prompt without KPIs does not include KPI header."""
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary)
        self.assertNotIn("KEY PERFORMANCE INDICATORS", prompt)

    def test_dataset_size_in_prompt(self):
        """Prompt reflects actual dataset size."""
        prompt = create_dataset_analysis_prompt(self.df, self.data_summary)
        self.assertIn(str(len(self.df)), prompt)


class TestVisualizationPrompt(unittest.TestCase):
    def setUp(self):
        self.column_metadata = {
            "sales": {"type": "numeric", "unique": 3},
            "category": {"type": "categorical", "unique": 2},
        }
        self.sample_data = "sales,category\n100,A\n150,B"

    def test_returns_string(self):
        """Prompt is a non-empty string."""
        prompt = create_visualization_prompt(self.column_metadata, self.sample_data)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_contains_json_format_hint(self):
        """Prompt instructs the model to return JSON."""
        prompt = create_visualization_prompt(self.column_metadata, self.sample_data)
        self.assertIn("json", prompt.lower())

    def test_with_kpis_adds_kpi_context(self):
        """KPI columns are mentioned in the prompt when supplied."""
        kpis = ['sales']
        prompt = create_visualization_prompt(self.column_metadata, self.sample_data, kpis)
        self.assertIn("sales", prompt)
        self.assertIn("key metrics", prompt.lower())

    def test_without_kpis_no_kpi_text(self):
        """Without KPIs the additional KPI context line is absent."""
        prompt = create_visualization_prompt(self.column_metadata, self.sample_data)
        self.assertNotIn("key metrics of interest", prompt.lower())

    def test_column_metadata_embedded_in_prompt(self):
        """Column metadata appears in the prompt body."""
        prompt = create_visualization_prompt(self.column_metadata, self.sample_data)
        self.assertIn("sales", prompt)
        self.assertIn("category", prompt)


class TestPatternExplanationPrompt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'sales': [100, 150, 120],
            'category': ['A', 'B', 'A']
        })

    def test_returns_string(self):
        """Prompt is a non-empty string."""
        prompt = create_pattern_explanation_prompt(self.df, "Sales spike on Tuesdays")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_pattern_description_in_prompt(self):
        """The pattern description appears verbatim in the prompt."""
        description = "Sales spike on Tuesdays"
        prompt = create_pattern_explanation_prompt(self.df, description)
        self.assertIn(description, prompt)

    def test_prompt_asks_for_explanations(self):
        """Prompt asks the model to explain the pattern."""
        prompt = create_pattern_explanation_prompt(self.df, "some pattern")
        self.assertIn("explanations", prompt.lower())


class TestAnalysisSummaryPrompt(unittest.TestCase):
    def setUp(self):
        self.analysis = "The dataset contains sales data for two categories."
        self.viz_specs = {
            "viz_1": {
                "type": "bar",
                "title": "Sales by Category",
                "description": "Shows sales distribution"
            }
        }

    def test_returns_string(self):
        """Prompt is a non-empty string."""
        prompt = create_analysis_summary_prompt(self.analysis, self.viz_specs)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_analysis_text_in_prompt(self):
        """The analysis text appears in the prompt."""
        prompt = create_analysis_summary_prompt(self.analysis, self.viz_specs)
        self.assertIn(self.analysis, prompt)

    def test_visualization_title_in_prompt(self):
        """Visualization titles appear in the prompt."""
        prompt = create_analysis_summary_prompt(self.analysis, self.viz_specs)
        self.assertIn("Sales by Category", prompt)

    def test_prompt_requests_insights(self):
        """Prompt asks the model to provide key insights."""
        prompt = create_analysis_summary_prompt(self.analysis, self.viz_specs)
        self.assertIn("insights", prompt.lower())


if __name__ == '__main__':
    unittest.main()
