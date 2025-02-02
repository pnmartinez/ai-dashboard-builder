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
    validity: float  # 0-1 score for technical correctness
    relevance: float  # 0-1 score for justification clarity
    usefulness: float  # 0-1 score for actionable insights
    diversity: float  # 0-1 score for visualization diversity
    redundancy: float  # 0-1 score for redundancy penalty
    
    def overall_score(self) -> float:
        # Weights based on the proposed architecture:
        # 60% for individual chart quality (split among validity, relevance, usefulness)
        # 30% for diversity
        # 10% for redundancy penalty
        individual_quality = (self.validity + self.relevance + self.usefulness) / 3
        
        overall = (
            0.6 * individual_quality +
            0.3 * self.diversity -
            0.1 * self.redundancy
        )
        return max(min(overall, 1.0), 0.0)  # Ensure score is between 0 and 1

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

        MIN_SAMPLE_SIZE = 2  # Minimum sample size for statistical tests
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
                'strength': 0.0
            }

            try:
                if col1_is_num and col2_is_num:
                    # Numeric-numeric: compute Pearson & Spearman
                    data1 = self.df[col1].dropna()
                    data2 = self.df[col2].dropna()
                    
                    # Check sample size
                    if len(data1) >= MIN_SAMPLE_SIZE and len(data2) >= MIN_SAMPLE_SIZE:
                        # Ensure both arrays have the same length
                        common_index = data1.index.intersection(data2.index)
                        if len(common_index) >= MIN_SAMPLE_SIZE:
                            data1 = data1[common_index]
                            data2 = data2[common_index]
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
                    
                    if col1_is_cat and col2_is_cat:
                        # Remove NaN values and ensure common indices
                        data = self.df[[col1, col2]].dropna()
                        if len(data) >= MIN_SAMPLE_SIZE:
                            contingency = pd.crosstab(data[col1], data[col2])
                            if contingency.size > 1:  # Ensure we have at least 2 categories
                                chi2, p, _, _ = chi2_contingency(contingency)
                                rel['type'] = 'cat-cat'
                                rel['stat'] = chi2
                                rel['p_value'] = p
                                rel['strength'] = chi2  # or a normalized measure
                    else:
                        # numeric-categorical
                        numeric_col = col1 if col1_is_num else col2
                        cat_col = col2 if col1_is_num else col1
                        data = self.df[[numeric_col, cat_col]].dropna()
                        
                        if len(data) >= MIN_SAMPLE_SIZE:
                            # Group data and check if we have enough samples per group
                            groups = [group for name, group in data.groupby(cat_col)[numeric_col] if len(group) >= MIN_SAMPLE_SIZE]
                            if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                                try:
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    if not np.isnan(f_stat):  # Check if the result is valid
                                        rel['type'] = 'numeric-cat'
                                        rel['stat'] = f_stat
                                        rel['p_value'] = p_val
                                        rel['strength'] = abs(f_stat)
                                except Exception as e:
                                    self.logger.debug(f"ANOVA failed: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error analyzing relationship between {col1} and {col2}: {str(e)}")

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

    def _find_relationship(self, col1: str, col2: str) -> Optional[Dict]:
        """Return the precomputed relationship dict for (col1, col2) if it exists."""
        for rel in self.relationships:
            if set(rel['columns']) == {col1, col2}:
                return rel
        return None

    def _evaluate_chart_validity(self, chart: Dict) -> float:
        """
        Evaluate if the visualization type is coherent with the data types.
        Returns a score between 0 and 1.
        """
        score = 1.0
        chart_type = VisualizationType(chart.get('type', ''))
        x, y = chart.get('x'), chart.get('y')
        
        # Check if columns exist
        if not x or x not in self.df.columns:
            return 0.0
        if chart_type not in [VisualizationType.HISTOGRAM, VisualizationType.PIE] and (not y or y not in self.df.columns):
            return 0.0

        # Evaluate based on chart type and data types
        if chart_type == VisualizationType.HISTOGRAM:
            if x not in self.numeric_cols:
                score *= 0.5
        elif chart_type == VisualizationType.LINE:
            # Line charts are valid for any ordered/continuous data
            # At least one axis should be numeric for meaningful trends
            if y not in self.numeric_cols:
                score *= 0.5
            # Small penalty if neither axis represents order (datetime or numeric)
            if x not in self.datetime_cols and x not in self.numeric_cols:
                score *= 0.9
        elif chart_type == VisualizationType.BAR:
            if x not in self.categorical_cols and y not in self.numeric_cols:
                score *= 0.7
            if x in self.numeric_cols and not chart.get('parameters', {}).get('aggregation'):
                score *= 0.8
        elif chart_type == VisualizationType.PIE:
            if x not in self.categorical_cols:
                score *= 0.5
            if x in self.categorical_cols and self.df[x].nunique() > 7:
                score *= 0.6
        elif chart_type == VisualizationType.SCATTER:
            if x not in self.numeric_cols or y not in self.numeric_cols:
                score *= 0.5
            # Check correlation if both numeric
            if x in self.numeric_cols and y in self.numeric_cols:
                rel = self._find_relationship(x, y)
                if rel and abs(rel['stat']) > 0.7:
                    desc = chart.get('description', '').lower()
                    if 'correlation' not in desc and 'relationship' not in desc:
                        score *= 0.9
        
        return score

    def _evaluate_chart_relevance(self, chart: Dict) -> float:
        """
        Evaluate the justification and relevance of the chart description.
        Uses a multi-faceted approach considering context, chart type, and content quality.
        Returns a score between 0 and 1.
        """
        description = chart.get('description', '').lower().strip()
        if not description:
            return 0.0
        
        score = 0.0
        words = description.split()
        
        # 1. Base Content Quality (30% of score)
        content_score = 0.0
        
        # Check for minimum meaningful content
        if len(words) >= 10:
            content_score += 0.15
        if len(words) >= 20:
            content_score += 0.15
            
        # Check for data-specific terms (column names, values)
        columns = [col for col in [chart.get('x'), chart.get('y'), chart.get('color')] if col]
        if any(col.lower() in description for col in columns):
            content_score += 0.3
            
        # Check for quantitative details
        has_numbers = any(char.isdigit() for char in description)
        if has_numbers:
            content_score += 0.2
            
        # Check for units or specific measures
        measurement_terms = ['percent', '%', 'average', 'mean', 'median', 'total', 'sum', 'count']
        if any(term in description for term in measurement_terms):
            content_score += 0.2
            
        score += min(content_score, 0.3)  # Cap at 30%
        
        # 2. Insight Type Relevance (40% of score)
        insight_score = 0.0
        chart_type = VisualizationType(chart.get('type', ''))
        
        # Define insight categories with weights and terms
        insight_categories = {
            'distribution': {
                'weight': 1.0,
                'terms': ['distribution', 'spread', 'range', 'variance', 'skew'],
                'relevant_types': [VisualizationType.HISTOGRAM, VisualizationType.BOX, VisualizationType.VIOLIN]
            },
            'trend': {
                'weight': 1.0,
                'terms': ['trend', 'pattern', 'evolution', 'over time', 'increase', 'decrease', 'change'],
                'relevant_types': [VisualizationType.LINE, VisualizationType.SCATTER]
            },
            'comparison': {
                'weight': 1.0,
                'terms': ['comparison', 'difference', 'versus', 'against', 'higher', 'lower', 'between'],
                'relevant_types': [VisualizationType.BAR, VisualizationType.SCATTER]
            },
            'correlation': {
                'weight': 1.0,
                'terms': ['correlation', 'relationship', 'association', 'connected', 'linked'],
                'relevant_types': [VisualizationType.SCATTER, VisualizationType.HEATMAP]
            },
            'composition': {
                'weight': 1.0,
                'terms': ['composition', 'breakdown', 'proportion', 'percentage', 'share', 'part'],
                'relevant_types': [VisualizationType.PIE, VisualizationType.BAR]
            }
        }
        
        matched_categories = 0
        total_weight = 0
        
        for category, info in insight_categories.items():
            # Higher weight if the insight type matches the chart type
            effective_weight = info['weight']
            if chart_type in info['relevant_types']:
                effective_weight *= 1.5
                
            if any(term in description for term in info['terms']):
                matched_categories += effective_weight
            total_weight += effective_weight
        
        if total_weight > 0:
            insight_score = matched_categories / total_weight
            score += min(insight_score * 0.4, 0.4)  # Cap at 40%
        
        # 3. Context and Relationship Quality (30% of score)
        context_score = 0.0
        
        # Check for relationship context if applicable
        if chart.get('x') and chart.get('y'):
            rel = self._find_relationship(chart.get('x'), chart.get('y'))
            if rel:
                # If there's a significant relationship, check if it's mentioned
                if rel['strength'] > 0.5:
                    relationship_terms = []
                    if rel['type'] == 'numeric-numeric':
                        relationship_terms = ['correlation', 'relationship', 'associated']
                    elif rel['type'] == 'numeric-cat':
                        relationship_terms = ['difference', 'varies', 'depends']
                    elif rel['type'] == 'cat-cat':
                        relationship_terms = ['association', 'linked', 'connected']
                        
                    if any(term in description for term in relationship_terms):
                        context_score += 0.15
        
        # Check for comparative context
        comparative_terms = ['more than', 'less than', 'compared to', 'relative to', 'whereas']
        if any(term in description for term in comparative_terms):
            context_score += 0.15
            
        # Check for causation/explanation attempts
        explanation_terms = ['because', 'due to', 'caused by', 'result of', 'leads to', 'suggests']
        if any(term in description for term in explanation_terms):
            context_score += 0.15
            
        # Check for business/domain relevance
        business_terms = ['business', 'market', 'customer', 'revenue', 'cost', 'performance', 'growth']
        if any(term in description for term in business_terms):
            context_score += 0.15
            
        score += min(context_score, 0.3)  # Cap at 30%
        
        return min(score, 1.0)  # Ensure final score is between 0 and 1

    def _evaluate_chart_usefulness(self, chart: Dict) -> float:
        """
        Evaluate if the visualization provides actionable insights.
        Returns a score between 0 and 1.
        """
        insight = chart.get('insight', '').strip()
        if not insight:
            return 0.0
        
        score = 0.0
        
        # Basic length check
        if len(insight) > 100:
            score += 0.4
        elif len(insight) > 50:
            score += 0.2
        
        # Check for quantitative information
        has_numbers = any(char.isdigit() for char in insight)
        if has_numbers:
            score += 0.3
            
        # Check for comparative language
        comparative_terms = ['more', 'less', 'higher', 'lower', 'increase', 'decrease', 'compared']
        if any(term in insight.lower() for term in comparative_terms):
            score += 0.3
            
        return min(score, 1.0)

    def _evaluate_dashboard_diversity(self, viz_specs: List[Dict]) -> float:
        """
        Evaluate if the dashboard covers different aspects of the dataset.
        Returns a score between 0 and 1.
        """
        # Track coverage of different aspects
        covered_types = set()  # data types covered
        covered_cols = set()   # columns used
        chart_types = set()    # visualization types used
        
        for spec in viz_specs:
            # Add columns and their types
            for col in [spec.get('x'), spec.get('y'), spec.get('color')]:
                if col:
                    covered_cols.add(col)
                    if col in self.numeric_cols:
                        covered_types.add('numeric')
                    elif col in self.categorical_cols:
                        covered_types.add('categorical')
                    elif col in self.datetime_cols:
                        covered_types.add('datetime')
            
            # Add chart type
            if spec.get('type'):
                chart_types.add(spec.get('type'))
        
        # Calculate scores for each aspect
        type_coverage = len(covered_types) / 3  # max 3 types (numeric, categorical, datetime)
        col_coverage = len(covered_cols) / len(self.df.columns)
        chart_variety = len(chart_types) / len(VisualizationType)
        
        # Combine scores with weights
        score = (0.4 * type_coverage + 0.3 * col_coverage + 0.3 * chart_variety)
        return score

    def _compute_redundancy_penalty(self, viz_specs: List[Dict]) -> float:
        """
        Calculate penalty for redundant visualizations.
        Returns a score between 0 and 1 where higher means more redundancy.
        """
        seen = {}
        penalty = 0.0
        total_charts = len(viz_specs)
        
        # First pass: count occurrences of each unique visualization
        for spec in viz_specs:
            # Create a key based on chart type and columns used
            key = (
                spec.get('type', ''),
                tuple(sorted([spec.get('x'), spec.get('y')]))
            )
            seen[key] = seen.get(key, 0) + 1
        
        # Calculate penalty based on duplicate counts
        for key, count in seen.items():
            if count > 1:
                # Add a penalty proportional to the number of duplicates
                # The penalty increases linearly with the number of duplicates
                duplicate_count = count - 1  # Number of charts beyond the first one
                penalty += 0.2 * duplicate_count / total_charts
            
            # Add smaller penalty for similar charts on same columns
            for other_key in seen:
                if key != other_key and key[1] == other_key[1]:  # Same columns, different chart type
                    penalty += 0.1 / total_charts
        
        return min(penalty, 1.0)  # Cap penalty at 1.0

    def evaluate_dashboard(self, viz_specs: List[Dict], dashboard_id: str = None) -> BenchmarkMetrics:
        """Evaluate the entire dashboard and return benchmark metrics."""
        self.logger.info("Beginning dashboard evaluation")
        
        # Store visualization-specific relationships if dashboard_id is provided
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

        # Chart Validity
        chart_validities = [self._evaluate_chart_validity(spec) for spec in viz_specs]
        validity = np.mean(chart_validities) if chart_validities else 0.5
        self.logger.info(f"Chart validity score: {validity:.2f}")

        # Chart Relevance
        chart_relevances = [self._evaluate_chart_relevance(spec) for spec in viz_specs]
        relevance = np.mean(chart_relevances) if chart_relevances else 0.5
        self.logger.info(f"Chart relevance score: {relevance:.2f}")

        # Chart Usefulness
        chart_usefulnesses = [self._evaluate_chart_usefulness(spec) for spec in viz_specs]
        usefulness = np.mean(chart_usefulnesses) if chart_usefulnesses else 0.5
        self.logger.info(f"Chart usefulness score: {usefulness:.2f}")

        # Dashboard Diversity
        diversity = self._evaluate_dashboard_diversity(viz_specs)
        self.logger.info(f"Dashboard diversity score: {diversity:.2f}")

        # Redundancy Penalty
        redundancy = self._compute_redundancy_penalty(viz_specs)
        self.logger.info(f"Redundancy penalty score: {redundancy:.2f}")
        
        metrics = BenchmarkMetrics(
            validity=validity,
            relevance=relevance,
            usefulness=usefulness,
            diversity=diversity,
            redundancy=redundancy
        )
        
        self.logger.info(f"Overall dashboard score: {metrics.overall_score():.2f}")
        return metrics
