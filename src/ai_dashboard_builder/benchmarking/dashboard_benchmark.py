"""Benchmark system for evaluating dashboard scientific validity and usefulness, 
   with expanded statistical checks and relationship coverage analysis."""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
import json
import os
from pathlib import Path

# Add fallback for when scipy is not available
try:
    import scipy.stats as stats
    from scipy.stats import pearsonr, spearmanr, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not installed - statistical validity checks will be limited")

class VisualizationType(Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    PIE = "pie"

@dataclass
class BenchmarkMetrics:
    statistical_validity: float  # 0-1 score
    visualization_appropriateness: float  # 0-1 score
    insight_value: float  # 0-1 score
    data_coverage: float  # 0-1 score
    
    def overall_score(self) -> float:
        # Adjust weights or turn this into a multi-objective approach as needed
        weights = {
            'statistical_validity': 0.4,
            'visualization_appropriateness': 0.3,
            'insight_value': 0.2,
            'data_coverage': 0.1
        }
        return (
            self.statistical_validity * weights['statistical_validity'] +
            self.visualization_appropriateness * weights['visualization_appropriateness'] +
            self.insight_value * weights['insight_value'] +
            self.data_coverage * weights['data_coverage']
        )

class DashboardBenchmark:
    def __init__(self, df: pd.DataFrame, output_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DashboardBenchmark")
        self.df = df
        self.output_dir = output_dir or "dashboard_analysis"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._analyze_data_types()
        # Precompute relationships between columns for deeper analysis
        self.relationships = self._analyze_relationships()
        # Store relationships in JSON
        self._store_relationships()

    def _analyze_data_types(self):
        """Analyze and store column types for reference."""
        self.logger.debug("Analyzing data types")
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        self.logger.debug(f"Found {len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical, and {len(self.datetime_cols)} datetime columns")

    def _analyze_relationships(self) -> List[Dict]:
        """Precompute interesting relationships (e.g., correlation, dependence) between columns."""
        self.logger.info("Analyzing relationships between columns")
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available - skipping relationship analysis")
            return []

        relationships = []
        cols = self.df.columns
        for col1, col2 in combinations(cols, 2):
            self.logger.debug(f"Analyzing relationship between {col1} and {col2}")
            col1_is_num = col1 in self.numeric_cols
            col2_is_num = col2 in self.numeric_cols
            rel = {
                'columns': (col1, col2),
                'type': None,
                'stat': None, 
                'p_value': None,
                'strength': 0.0  # you could use absolute correlation or mutual info, etc.
            }

            try:
                if col1_is_num and col2_is_num:
                    # Numeric-numeric: compute Pearson & Spearman
                    data1 = self.df[col1].dropna()
                    data2 = self.df[col2].dropna()

                    if len(data1) > 1 and len(data2) > 1:
                        pearson_r, pearson_p = pearsonr(data1, data2)
                        spearman_r, spearman_p = spearmanr(data1, data2)
                        self.logger.debug(f"Pearson correlation: {pearson_r:.3f} (p={pearson_p:.3f})")
                        rel['type'] = 'numeric-numeric'
                        rel['stat'] = pearson_r
                        rel['p_value'] = pearson_p
                        rel['strength'] = abs(pearson_r)
                else:
                    # At least one is categorical
                    col1_is_cat = col1 in self.categorical_cols
                    col2_is_cat = col2 in self.categorical_cols
                    # For numeric-categorical: can do ANOVA, or compare means across categories, etc.
                    # For cat-cat: can do chi-square
                    if col1_is_cat and col2_is_cat:
                        contingency = pd.crosstab(self.df[col1], self.df[col2])
                        chi2, p, _, _ = chi2_contingency(contingency)
                        rel['type'] = 'cat-cat'
                        rel['stat'] = chi2
                        rel['p_value'] = p
                        rel['strength'] = chi2  # or a normalized measure
                    else:
                        # numeric-categorical: one approach is ANOVA or Kruskal-Wallis
                        numeric_col = col1 if col1_is_num else col2
                        cat_col = col2 if col1_is_num else col1
                        data = self.df[[numeric_col, cat_col]].dropna()
                        if data[cat_col].nunique() > 1:
                            # Example: one-way ANOVA
                            groups = [d for _, d in data.groupby(cat_col)[numeric_col]]
                            if len(groups) > 1:
                                try:
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    rel['type'] = 'numeric-cat'
                                    rel['stat'] = f_stat
                                    rel['p_value'] = p_val
                                    rel['strength'] = abs(f_stat)
                                except:
                                    pass

            except Exception as e:
                self.logger.error(f"Error analyzing relationship between {col1} and {col2}: {str(e)}")

            # You could do a multiple testing correction across all pairs
            relationships.append(rel)
        
        self.logger.info(f"Completed relationship analysis. Found {len(relationships)} relationships")
        return relationships

    def _store_relationships(self):
        """Store relationships data in a JSON file."""
        self.logger.info("Storing relationships data")
        relationships_data = []
        
        for rel in self.relationships:
            # Convert numpy types to native Python types for JSON serialization
            clean_rel = {
                'columns': rel['columns'],
                'type': rel['type'],
                'stat': float(rel['stat']) if rel['stat'] is not None else None,
                'p_value': float(rel['p_value']) if rel['p_value'] is not None else None,
                'strength': float(rel['strength'])
            }
            relationships_data.append(clean_rel)
            
        output_path = os.path.join(self.output_dir, 'relationships.json')
        with open(output_path, 'w') as f:
            json.dump(relationships_data, f, indent=2)
        self.logger.info(f"Stored relationships data in {output_path}")

    def _evaluate_statistical_validity(self, viz_spec: Dict) -> float:
        """Evaluate statistical validity of visualization with deeper checks."""
        self.logger.debug(f"Evaluating statistical validity for visualization: {viz_spec['type']}")
        score = 1.0
        viz_type = VisualizationType(viz_spec['type'])
        x, y = viz_spec.get('x'), viz_spec.get('y')

        # If numeric-numeric, check if we have correlation data
        if x in self.numeric_cols and y in self.numeric_cols:
            # Look up the precomputed relationship
            rel = self._find_relationship(x, y)
            if rel and SCIPY_AVAILABLE:
                # If distribution is strongly non-normal, using Pearson might be suboptimal
                # We'll do a quick normality check again for demonstration
                try:
                    _, p_value_x = stats.shapiro(self.df[x].dropna())
                    _, p_value_y = stats.shapiro(self.df[y].dropna())
                    if p_value_x < 0.05 or p_value_y < 0.05:
                        # If the chart claims Pearson correlation explicitly, penalize
                        if 'pearson' in viz_spec.get('description', '').lower():
                            score *= 0.8
                except:
                    pass

                # If there's a strong correlation but the chart doesn't mention it, we might penalize
                # or if the chart is scatter with no mention of correlation, we might just do a small penalty
                if viz_type == VisualizationType.SCATTER and abs(rel['stat']) > 0.7:
                    desc = viz_spec.get('description', '').lower()
                    if 'correlation' not in desc and 'relationship' not in desc:
                        score *= 0.9  # missed an opportunity to mention strong relationship

        # For numeric-categorical, if using bar chart, check if an aggregation is present
        if viz_type == VisualizationType.BAR:
            if x in self.numeric_cols and not viz_spec.get('parameters', {}).get('aggregation'):
                score *= 0.8  # Penalty for using bar plot with raw numeric data

        # For pie charts with many categories
        if viz_type == VisualizationType.PIE and x in self.categorical_cols:
            n_categories = self.df[x].nunique()
            if n_categories > 7:
                score *= 0.6  # Penalty for pie chart with too many categories

        self.logger.debug(f"Statistical validity score: {score}")
        return score

    def _find_relationship(self, col1: str, col2: str) -> Optional[Dict]:
        """Return the precomputed relationship dict for (col1, col2) if it exists."""
        for rel in self.relationships:
            if set(rel['columns']) == {col1, col2}:
                return rel
        return None

    def _evaluate_visualization_appropriateness(self, viz_spec: Dict) -> float:
        """Evaluate if visualization type is appropriate for the data."""
        score = 1.0
        viz_type = VisualizationType(viz_spec['type'])
        x, y = viz_spec.get('x'), viz_spec.get('y')
        
        # Example: line chart for time series
        if viz_type == VisualizationType.LINE:
            if x not in self.datetime_cols and y not in self.datetime_cols:
                score *= 0.7  # penalty for line plot without time dimension

        # Example: histogram for numeric data
        if viz_type == VisualizationType.HISTOGRAM and x not in self.numeric_cols:
            score *= 0.5

        return score

    def _evaluate_relationship_coverage(self, viz_specs: List[Dict]) -> float:
        """
        Check if the strongest relationships in the dataset are actually visualized 
        (and visualized with an appropriate chart). 
        Return a coverage ratio or insight-like score in [0..1].
        """
        self.logger.debug("Evaluating relationship coverage")
        if not self.relationships:
            self.logger.warning("No relationships available for coverage evaluation")
            return 0.5

        # 1) Filter to top relationships by 'strength'
        #    For demonstration, let's define "interesting" as top 5 with strength > threshold
        sorted_rels = sorted(self.relationships, key=lambda r: r['strength'], reverse=True)
        interesting_rels = [r for r in sorted_rels if r['strength'] > 0.3][:5]
        self.logger.debug(f"Interesting relationships: {interesting_rels}")

        # 2) Check how many of these interesting relationships are visualized
        covered = 0
        for rel in interesting_rels:
            (c1, c2) = rel['columns']
            found_viz = False
            appropriate_viz = False
            for spec in viz_specs:
                x, y = spec.get('x'), spec.get('y')
                if {x, y} == {c1, c2}:
                    found_viz = True
                    # Check if the chart type is appropriate
                    if rel['type'] == 'numeric-numeric':
                        # Scatter or possibly line if one is time
                        if spec['type'] in [VisualizationType.SCATTER.value, VisualizationType.LINE.value]:
                            appropriate_viz = True
                    elif rel['type'] == 'numeric-cat':
                        # Box plot, violin, or bar with aggregation might be good
                        if spec['type'] in [VisualizationType.BOX.value, VisualizationType.VIOLIN.value]:
                            appropriate_viz = True
                    elif rel['type'] == 'cat-cat':
                        # Stacked bar or heatmap might be typical
                        if spec['type'] in [VisualizationType.HEATMAP.value, VisualizationType.BAR.value]:
                            appropriate_viz = True
                    if appropriate_viz:
                        covered += 1
                    break
            # If we found a relevant viz but it wasn't truly appropriate, you might do partial credit

        coverage_ratio = covered / len(interesting_rels) if interesting_rels else 1.0
        self.logger.info(f"Relationship coverage score: {coverage_ratio:.2f}")
        return coverage_ratio

    def _evaluate_insight_value(self, viz_specs: List[Dict]) -> float:
        """Evaluate the potential insight value. Incorporate relationship coverage."""
        base_scores = []
        for spec in viz_specs:
            score = 1.0
            description = spec.get('description', '').lower()
            # Check for meaningful analysis in description
            if 'trend' in description or 'pattern' in description:
                score *= 1.1
            if 'correlation' in description or 'relationship' in description:
                score *= 1.1
            if 'outlier' in description or 'anomaly' in description:
                score *= 1.1
            base_scores.append(min(score, 1.0))

        # Average textual/heuristic insight
        if base_scores:
            textual_insight = np.mean(base_scores)
        else:
            textual_insight = 0.5

        # Relationship coverage: how many strong relationships are properly shown
        rel_coverage = self._evaluate_relationship_coverage(viz_specs)

        # Combine them: you could weigh them differently. Example:
        # 70% textual insight, 30% coverage
        combined_insight = 0.7 * textual_insight + 0.3 * rel_coverage
        return combined_insight

    def _evaluate_data_coverage(self, viz_specs: List[Dict]) -> float:
        """Evaluate how well the visualizations cover columns (basic coverage)."""
        covered_cols = set()
        for spec in viz_specs:
            covered_cols.add(spec.get('x'))
            covered_cols.add(spec.get('y'))
            covered_cols.add(spec.get('color'))
        
        covered_cols.discard(None)
        # Basic ratio
        coverage_ratio = len(covered_cols) / len(self.df.columns)
        return min(coverage_ratio, 1.0)

    def evaluate_dashboard(self, viz_specs: List[Dict], dashboard_id: str = None) -> BenchmarkMetrics:
        """Evaluate the entire dashboard and return benchmark metrics."""
        self.logger.info("Beginning dashboard evaluation")
        
        # Store visualization-specific relationships
        if dashboard_id:
            viz_relationships = []
            for i, spec in enumerate(viz_specs):
                x, y = spec.get('x'), spec.get('y')
                if x and y:
                    rel = self._find_relationship(x, y)
                    if rel:
                        viz_relationships.append({
                            'viz_index': i,
                            'relationship': {
                                'columns': rel['columns'],
                                'type': rel['type'],
                                'stat': float(rel['stat']) if rel['stat'] is not None else None,
                                'p_value': float(rel['p_value']) if rel['p_value'] is not None else None,
                                'strength': float(rel['strength'])
                            }
                        })
            
            # Store visualization-specific relationships
            viz_rel_path = os.path.join(self.output_dir, f'viz_relationships_{dashboard_id}.json')
            with open(viz_rel_path, 'w') as f:
                json.dump(viz_relationships, f, indent=2)
            self.logger.info(f"Stored visualization relationships in {viz_rel_path}")

        # Statistical Validity
        stat_validities = [self._evaluate_statistical_validity(spec) for spec in viz_specs]
        statistical_validity = np.mean(stat_validities) if stat_validities else 0.5
        self.logger.info(f"Statistical validity score: {statistical_validity:.2f}")

        # Visualization Appropriateness
        viz_appropriatenesses = [self._evaluate_visualization_appropriateness(spec) for spec in viz_specs]
        visualization_appropriateness = np.mean(viz_appropriatenesses) if viz_appropriatenesses else 0.5
        self.logger.info(f"Visualization appropriateness score: {visualization_appropriateness:.2f}")
        
        # Insight Value
        insight_value = self._evaluate_insight_value(viz_specs)
        self.logger.info(f"Insight value score: {insight_value:.2f}")
        
        # Data Coverage
        data_coverage = self._evaluate_data_coverage(viz_specs)
        self.logger.info(f"Data coverage score: {data_coverage:.2f}")
        
        metrics = BenchmarkMetrics(
            statistical_validity=statistical_validity,
            visualization_appropriateness=visualization_appropriateness,
            insight_value=insight_value,
            data_coverage=data_coverage
        )
        
        self.logger.info(f"Overall dashboard score: {metrics.overall_score():.2f}")
        return metrics
