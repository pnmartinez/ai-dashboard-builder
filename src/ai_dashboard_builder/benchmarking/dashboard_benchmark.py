"""Benchmark system for evaluating dashboard scientific validity and usefulness."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

# Add fallback for when scipy is not available
try:
    import scipy.stats as stats
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
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._analyze_data_types()
        
    def _analyze_data_types(self):
        """Analyze and store column types for reference."""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        
    def _evaluate_statistical_validity(self, viz_spec: Dict) -> float:
        """Evaluate statistical validity of visualization."""
        score = 1.0
        viz_type = VisualizationType(viz_spec['type'])
        x, y = viz_spec.get('x'), viz_spec.get('y')
        
        if viz_type == VisualizationType.SCATTER:
            # Check if correlation analysis is appropriate
            if x in self.numeric_cols and y in self.numeric_cols:
                if SCIPY_AVAILABLE:
                    # Perform Shapiro-Wilk test for normality
                    try:
                        _, p_value_x = stats.shapiro(self.df[x].dropna())
                        _, p_value_y = stats.shapiro(self.df[y].dropna())
                        
                        if p_value_x < 0.05 or p_value_y < 0.05:
                            # Non-normal distribution - should use Spearman
                            if 'correlation' in viz_spec.get('description', '').lower():
                                if 'pearson' in viz_spec.get('description', '').lower():
                                    score *= 0.7  # Penalty for using Pearson with non-normal data
                    except Exception:
                        # Fallback if statistical test fails
                        pass
                else:
                    # Basic check without scipy
                    if 'correlation' in viz_spec.get('description', '').lower():
                        score *= 0.9  # Small penalty since we can't verify normality
                
        elif viz_type == VisualizationType.BAR:
            # Check if bar plot is appropriate for the data type
            if x in self.numeric_cols and not viz_spec.get('parameters', {}).get('aggregation'):
                score *= 0.8  # Penalty for using bar plot with non-aggregated numeric data
                
        elif viz_type == VisualizationType.PIE:
            # Check if pie chart is appropriate (not too many categories)
            if x in self.categorical_cols:
                n_categories = self.df[x].nunique()
                if n_categories > 7:
                    score *= 0.6  # Penalty for pie chart with too many categories
                    
        return score

    def _evaluate_visualization_appropriateness(self, viz_spec: Dict) -> float:
        """Evaluate if visualization type is appropriate for the data."""
        score = 1.0
        viz_type = VisualizationType(viz_spec['type'])
        x, y = viz_spec.get('x'), viz_spec.get('y')
        
        # Check data distribution vs visualization type
        if viz_type == VisualizationType.LINE:
            if not any(x == col for col in self.datetime_cols):
                score *= 0.7  # Penalty for line plot without time series
                
        elif viz_type == VisualizationType.HISTOGRAM:
            if x not in self.numeric_cols:
                score *= 0.5  # Major penalty for histogram with non-numeric data
                
        return score

    def _evaluate_insight_value(self, viz_spec: Dict) -> float:
        """Evaluate the potential insight value of the visualization."""
        score = 1.0
        description = viz_spec.get('description', '').lower()
        
        # Check for meaningful analysis in description
        if 'trend' in description or 'pattern' in description:
            score *= 1.1  # Bonus for trend analysis
        if 'correlation' in description or 'relationship' in description:
            score *= 1.1  # Bonus for relationship analysis
        if 'outlier' in description or 'anomaly' in description:
            score *= 1.1  # Bonus for outlier detection
            
        return min(score, 1.0)  # Cap at 1.0

    def _evaluate_data_coverage(self, viz_specs: List[Dict]) -> float:
        """Evaluate how well the visualizations cover the important aspects of the dataset."""
        covered_cols = set()
        for spec in viz_specs:
            covered_cols.add(spec.get('x'))
            covered_cols.add(spec.get('y'))
            covered_cols.add(spec.get('color'))
        
        covered_cols.discard(None)
        coverage_ratio = len(covered_cols) / len(self.df.columns)
        return coverage_ratio

    def evaluate_dashboard(self, viz_specs: List[Dict]) -> BenchmarkMetrics:
        """Evaluate the entire dashboard and return benchmark metrics."""
        statistical_validity = np.mean([
            self._evaluate_statistical_validity(spec) for spec in viz_specs
        ])
        
        visualization_appropriateness = np.mean([
            self._evaluate_visualization_appropriateness(spec) for spec in viz_specs
        ])
        
        insight_value = np.mean([
            self._evaluate_insight_value(spec) for spec in viz_specs
        ])
        
        data_coverage = self._evaluate_data_coverage(viz_specs)
        
        return BenchmarkMetrics(
            statistical_validity=statistical_validity,
            visualization_appropriateness=visualization_appropriateness,
            insight_value=insight_value,
            data_coverage=data_coverage
        )