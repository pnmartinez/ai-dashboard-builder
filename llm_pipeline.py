import os
from typing import Dict, Any, Optional
import pandas as pd
import requests
import json
import logging
import numpy as np
import time
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LLMPipeline')
logger.setLevel(logging.INFO)

class LLMPipeline:
    def __init__(self, model_name: str = "mistral", use_local: bool = True):
        self.model_name = model_name
        self.use_local = use_local
        # Get Ollama host from environment variable or default to host.docker.internal
        ollama_host = os.getenv('OLLAMA_HOST', 'host.docker.internal')
        self.ollama_base_url = f"http://{ollama_host}:11434/api"
        print(self.ollama_base_url)
        
        # Create responses directory
        self.responses_dir = "llm_responses"
        if not os.path.exists(self.responses_dir):
            os.makedirs(self.responses_dir)
            
        logger.info(f"Initializing LLMPipeline with model: {model_name} (local: {use_local})")
        
        if not use_local:
            self.api_key = os.getenv("LLM_API_KEY")
            if not self.api_key:
                logger.error("API key not found in environment variables")
                raise ValueError("API key not found in environment variables")

    def _serialize_for_json(self, obj):
        """Helper method to make data JSON serializable"""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def _time_execution(self, func_name: str, func, *args, **kwargs):
        """Helper method to time function execution"""
        start_time = time.time()
        logger.info(f"Starting {func_name}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {func_name} in {duration:.2f} seconds")
        return result

    def _save_interaction(self, prompt: str, response: str, metadata: dict = None):
        """Save LLM interaction to disk"""
        try:
            # Create a unique filename based on timestamp and prompt hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"{timestamp}_{prompt_hash}.json"
            
            # Prepare interaction data
            interaction_data = {
                "timestamp": timestamp,
                "model": self.model_name,
                "provider": "local" if self.use_local else "api",
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {}
            }
            
            # Save to file
            filepath = os.path.join(self.responses_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved interaction to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save interaction: {str(e)}", exc_info=True)

    def _query_local(self, prompt: str) -> str:
        logger.info(f"Querying local Ollama model: {self.model_name}")
        try:
            start_time = time.time()
            logger.info("Sending request to Ollama API")
            response = requests.post(
                f"{self.ollama_base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            logger.info(f"Received response from Ollama (status: {response.status_code})")
            
            if not response.ok:
                error_msg = f"Ollama API returned status code {response.status_code}"
                logger.error(error_msg)
                self._save_interaction(prompt, f"Error: {error_msg}", {
                    "status": "error",
                    "error_type": "http_error",
                    "status_code": response.status_code
                })
                return f"Error: {error_msg}"
                
            response_json = response.json()
            
            if 'response' not in response_json:
                error_msg = f"Unexpected response format: {response_json}"
                logger.error(error_msg)
                self._save_interaction(prompt, f"Error: {error_msg}", {
                    "status": "error",
                    "error_type": "format_error",
                    "raw_response": response_json
                })
                raise ValueError(error_msg)
                
            response_text = response_json["response"]
            duration = time.time() - start_time
            
            # Save successful interaction
            self._save_interaction(prompt, response_text, {
                "status": "success",
                "duration_seconds": duration
            })
            
            logger.info("Successfully processed response from local model")
            return response_text
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error querying local model: {error_msg}", exc_info=True)
            self._save_interaction(prompt, f"Error: {error_msg}", {
                "status": "error",
                "error_type": "exception",
                "error_message": error_msg
            })
            return f"Error: {error_msg}"

    def _query_api(self, prompt: str) -> str:
        """Query external API based on selected model."""
        start_time = time.time()
        try:
            response_text = ""
            
            if 'gpt' in self.model_name:
                # OpenAI GPT API (>=1.0.0)
                from openai import OpenAI
                
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                response_text = response.choices[0].message.content
                
            elif 'claude' in self.model_name:
                # Anthropic Claude API
                headers = {
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000
                    }
                )
                response_json = response.json()
                response_text = response_json['content'][0]['text']
                
            elif 'mistral' in self.model_name:
                # Mistral AI API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                )
                response_json = response.json()
                response_text = response_json['choices'][0]['message']['content']
            
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            duration = time.time() - start_time
            
            # Save successful interaction
            self._save_interaction(prompt, response_text, {
                "status": "success",
                "duration_seconds": duration,
                "model": self.model_name
            })
            
            return response_text
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error querying {self.model_name} API: {error_msg}", exc_info=True)
            
            # Save failed interaction
            self._save_interaction(prompt, f"Error: {error_msg}", {
                "status": "error",
                "error_type": "api_error",
                "error_message": error_msg,
                "duration_seconds": duration,
                "model": self.model_name
            })
            
            return f"Error: {error_msg}"

    def analyze_dataset(self, df: pd.DataFrame) -> str:
        def _analyze():
            logger.info("Starting dataset analysis")
            try:
                # Create dataset summary
                data_summary = {
                    "columns": list(df.columns),
                    "sample_rows": [{k: self._serialize_for_json(v) for k, v in row.items()} 
                                  for row in df.head(5).to_dict('records')],
                    "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                    "null_counts": {k: self._serialize_for_json(v) for k, v in df.isnull().sum().to_dict().items()},
                    "unique_counts": {k: self._serialize_for_json(v) for k, v in df.nunique().to_dict().items()}
                }
                
                # Generate and send prompt
                prompt = f"""Analyze this dataset and provide a detailed report in the following format:

📊 DATASET OVERVIEW
------------------
• Total Records: {len(df)}
• Total Features: {len(df.columns)}
• Time Period: [infer from data if applicable]
• Dataset Purpose: [infer from content]

📋 COLUMN ANALYSIS
-----------------
{", ".join(data_summary['columns'])}

For each column:
• Type: [data type]
• Description: [what this column represents]
• Value Range/Categories: [key values or ranges]
• Quality Issues: [missing values, anomalies]

🔍 KEY OBSERVATIONS
-----------------
• [List 3-5 main patterns or insights]
• [Note any data quality issues]
• [Highlight interesting relationships]

📈 STATISTICAL HIGHLIGHTS
-----------------------
• [Key statistics and distributions]
• [Notable correlations]
• [Significant patterns]

💡 RECOMMENDATIONS
----------------
• [Suggest data cleaning steps]
• [Propose analysis approaches]
• [Recommend focus areas]

Sample Data Preview:
{pd.DataFrame(data_summary['sample_rows']).to_string()}

Additional Information:
- Data Types: {data_summary['data_types']}
- Unique Values: {data_summary['unique_counts']}
- Null Counts: {data_summary['null_counts']}

Please provide a comprehensive analysis following this exact structure, using the section headers and emoji markers as shown."""
                
                # Query model and save response
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)
                
                return response
                
            except Exception as e:
                logger.error(f"Error in analyze_dataset: {str(e)}", exc_info=True)
                raise
        return self._time_execution("analyze_dataset", _analyze)

    def suggest_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        def _suggest():
            logger.info("Starting visualization suggestions process")
            try:
                # Get column metadata
                logger.info("Creating column metadata")
                column_metadata = {
                    col: {
                        'dtype': str(df[col].dtype),
                        'unique_count': self._serialize_for_json(df[col].nunique()),
                        'null_count': self._serialize_for_json(df[col].isnull().sum()),
                        'is_temporal': pd.api.types.is_datetime64_any_dtype(df[col]),
                        'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                        'is_categorical': pd.api.types.is_categorical_dtype(df[col]) or 
                                        (df[col].nunique() / len(df) < 0.05),
                        'sample_values': [self._serialize_for_json(v) for v in df[col].head().tolist()]
                    } for col in df.columns
                }
                
                logger.info("Column metadata created successfully")

                # Generate the prompt
                column_metadata_json = json.dumps(column_metadata, indent=2)
                sample_data = df.head(3).to_string()
                example_format = """
{
    "viz_1": {
        "type": "bar",
        "x": "day_of_week",
        "y": "count",
        "color": "matched_keyword",
        "title": "Event Distribution by Day",
        "description": "Shows event frequency across days",
        "parameters": {
            "orientation": "v",
            "aggregation": "count"
        }
    }
}"""

                prompt = f"""Given the following dataset information, suggest appropriate visualizations in a structured format.

Column Metadata:
{column_metadata_json}

Sample data:
{sample_data}

Return a JSON structure where each key is a visualization ID and the value contains:
1. type: The chart type (e.g., 'line', 'bar', 'scatter', 'histogram', 'box', 'heatmap')
2. x: Column(s) for x-axis
3. y: Column(s) for y-axis (if applicable)
4. color: Column for color encoding (if applicable)
5. title: Suggested title for the visualization
6. description: What insights this visualization provides
7. parameters: Additional parameters for the chart (e.g., orientation, aggregation)

Example format:
{example_format}"""

                logger.info("Generated visualization prompt")

                # Query model and save response
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)
                    
                logger.info("Received visualization response")

                # Extract JSON from the response
                logger.info("Extracting JSON from response")
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    logger.error("Could not find JSON structure in response")
                    return {}
                    
                json_str = response[start_idx:end_idx]
                logger.info("Found JSON structure in response")

                # Parse the JSON
                viz_specs = json.loads(json_str)
                logger.info(f"Successfully parsed JSON with {len(viz_specs)} visualization specifications")

                # Add metadata to each visualization
                logger.info("Adding metadata to visualization specifications")
                for viz_id, viz_spec in viz_specs.items():
                    viz_spec['metadata'] = {
                        'x_meta': column_metadata.get(viz_spec.get('x', ''), {}),
                        'y_meta': column_metadata.get(viz_spec.get('y', ''), {}),
                        'color_meta': column_metadata.get(viz_spec.get('color', ''), {})
                    }
                    
                    viz_spec['parameters'] = viz_spec.get('parameters', {})
                    viz_spec['parameters'].update({
                        'height': 400,
                        'template': 'plotly',
                        'opacity': 0.8
                    })

                # Save visualization specifications separately
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_specs_file = os.path.join(self.responses_dir, f"viz_specs_{timestamp}.json")
                try:
                    with open(viz_specs_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "timestamp": timestamp,
                            "model": self.model_name,
                            "provider": "local" if self.use_local else "api",
                            "column_metadata": column_metadata,
                            "visualization_specs": viz_specs
                        }, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved visualization specifications to {viz_specs_file}")
                except Exception as e:
                    logger.error(f"Failed to save visualization specifications: {str(e)}", exc_info=True)

                logger.info("Successfully generated all visualization specifications")
                return viz_specs
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}", exc_info=True)
                return {}
            except Exception as e:
                logger.error(f"Error in visualization suggestion: {str(e)}", exc_info=True)
                return {}
        return self._time_execution("suggest_visualizations", _suggest)

    def explain_pattern(self, df: pd.DataFrame, pattern_description: str) -> str:
        """
        Get LLM explanation for a specific pattern in the data.
        
        Args:
            df (pd.DataFrame): The dataset
            pattern_description (str): Description of the pattern to analyze
            
        Returns:
            str: Pattern explanation
        """
        prompt = f"""Analyze the following pattern in the dataset:
{pattern_description}

Dataset sample:
{df.head(5).to_string()}

Please provide:
1. Possible explanations for this pattern
2. Whether this pattern is expected or anomalous
3. Potential implications or impacts
4. Recommendations for further investigation
"""
        
        if self.use_local:
            return self._query_local(prompt)
        else:
            return self._query_api(prompt)

    def summarize_analysis(self, analysis: str, viz_specs: Dict[str, Any]) -> str:
        """
        Create a final summary of the analysis and visualizations.
        
        Args:
            analysis (str): The dataset analysis text
            viz_specs (Dict[str, Any]): The visualization specifications
            
        Returns:
            str: A concise summary of key findings and insights
        """
        def _summarize():
            logger.info("Starting analysis summarization")
            try:
                # Create the prompt with analysis and visualization info
                prompt = f"""Based on the following dataset analysis and visualizations created, provide a concise summary of the key findings and insights.

Dataset Analysis:
{analysis}

Visualizations Created:
{json.dumps([{
    'title': viz['title'],
    'type': viz['type'],
    'description': viz['description']
} for viz in viz_specs.values()], indent=2)}

Please provide:
1. A brief overview of the dataset's purpose and content
2. 3-5 key insights discovered from the analysis
3. The most important patterns or trends shown in the visualizations
4. Any potential recommendations or next steps for further analysis

Keep the summary concise and focused on actionable insights."""

                # Query model
                logger.info("Querying model for summary")
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)
                
                logger.info("Successfully generated analysis summary")
                return response
                
            except Exception as e:
                logger.error(f"Error in analysis summarization: {str(e)}", exc_info=True)
                return "Error generating summary: " + str(e)
                
        return self._time_execution("summarize_analysis", _summarize)

# Example usage:
if __name__ == "__main__":
    # Example with local Ollama
    pipeline = LLMPipeline(model_name="mistral", use_local=True)
    
    # Load your dataset
    df = pd.read_csv("your_dataset.csv")  # Replace with your data loading
    
    # Get analysis
    analysis = pipeline.analyze_dataset(df)
    print(analysis) 