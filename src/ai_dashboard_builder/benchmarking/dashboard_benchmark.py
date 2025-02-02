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
from scipy import stats

# Add fallback for when scipy is not available
try:
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
        self.relationship_cache = {}  # Initialize relationship cache
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._analyze_data_types()
        self.relationships = []  # Initialize relationships list
        self._analyze_relationships()  # This will populate self.relationships
        self._store_relationships()

    def _analyze_data_types(self):
        """Analyze and store column types for reference."""
        self.logger.debug("Analyzing data types")
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        self.logger.debug(f"Found {len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical, and {len(self.datetime_cols)} datetime columns")

    def _convert_to_native(self, value):
        """Convert numpy types to native Python types."""
        if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(value)
        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _find_relationship(self, var1: str, var2: str) -> Dict:
        """Find the statistical relationship between two variables."""
        try:
            # Get relationship info from cache if available
            cache_key = f"{var1}_{var2}"
            if cache_key in self.relationship_cache:
                return {k: self._convert_to_native(v) for k, v in self.relationship_cache[cache_key].items()}

            # If both variables are numeric
            if var1 in self.numeric_cols and var2 in self.numeric_cols:
                correlation, p_value = pearsonr(
                    self.df[var1].fillna(0),
                    self.df[var2].fillna(0)
                )
                relationship = {
                    'type': 'numeric-numeric',
                    'stat': self._convert_to_native(correlation),
                    'p_value': self._convert_to_native(p_value)
                }

            # If both variables are categorical
            elif var1 in self.categorical_cols and var2 in self.categorical_cols:
                contingency = pd.crosstab(self.df[var1], self.df[var2])
                chi2, p_value, _, _ = chi2_contingency(contingency)
                relationship = {
                    'type': 'cat-cat',
                    'stat': self._convert_to_native(chi2),
                    'p_value': self._convert_to_native(p_value)
                }

            # If one variable is numeric and the other is categorical
            elif (var1 in self.numeric_cols and var2 in self.categorical_cols) or \
                 (var2 in self.numeric_cols and var1 in self.categorical_cols):
                num_var = var1 if var1 in self.numeric_cols else var2
                cat_var = var2 if var1 in self.numeric_cols else var1
                
                # Perform one-way ANOVA
                categories = self.df[cat_var].unique()
                samples = [self.df[self.df[cat_var] == cat][num_var].dropna() for cat in categories]
                f_stat, p_value = stats.f_oneway(*samples) if len(samples) > 1 else (0, 1)
                
                relationship = {
                    'type': 'numeric-cat',
                    'stat': self._convert_to_native(f_stat),
                    'p_value': self._convert_to_native(p_value)
                }
            else:
                return None

            # Cache the result
            self.relationship_cache[cache_key] = relationship
            return relationship

        except Exception as e:
            self.logger.error(f"Error finding relationship between {var1} and {var2}: {str(e)}")
            return None

    def _analyze_relationships(self):
        """Analyze relationships between all pairs of columns."""
        self.logger.info("Analyzing relationships between columns")
        
        # Get all columns to analyze
        columns = list(self.numeric_cols) + list(self.categorical_cols)
        
        # Find relationships between all pairs
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                rel = self._find_relationship(col1, col2)
                if rel:
                    self.relationships.append({
                        'var1': col1,
                        'var2': col2,
                        'relationship': rel
                    })
        
        self.logger.info(f"Completed relationship analysis. Found {len(self.relationships)} relationships")
        
        # Store relationships data
        self.logger.info("Storing relationships data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert all numpy types to native Python types before serialization
        def convert_relationships(rels):
            converted = []
            for rel in rels:
                converted.append({
                    'var1': rel['var1'],
                    'var2': rel['var2'],
                    'relationship': {
                        k: self._convert_to_native(v) for k, v in rel['relationship'].items()
                    }
                })
            return converted
        
        with open(os.path.join(self.output_dir, 'relationships.json'), 'w') as f:
            json.dump(convert_relationships(self.relationships), f, indent=2)
        
        self.logger.info(f"Stored relationships data in {os.path.join(self.output_dir, 'relationships.json')}")

    def _store_relationships(self):
        """Store relationships data in a JSON file."""
        if not hasattr(self, 'relationships') or not self.relationships:
            self.logger.warning("No relationships to store")
            return

        self.logger.info("Storing relationships data")
        relationships_data = []
        
        for rel in self.relationships:
            # Convert numpy types to native Python types for JSON serialization
            clean_rel = {
                'var1': rel['var1'],
                'var2': rel['var2'],
                'relationship': {
                    k: self._convert_to_native(v) for k, v in rel['relationship'].items()
                }
            }
            relationships_data.append(clean_rel)
            
        output_path = os.path.join(self.output_dir, 'relationships.json')
        with open(output_path, 'w') as f:
            json.dump(relationships_data, f, indent=2)
        self.logger.info(f"Stored relationships data in {output_path}")

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
                                'var1': rel['var1'],
                                'var2': rel['var2'],
                                'type': rel['type'],
                                'stat': rel['stat'],
                                'p_value': rel['p_value']
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
