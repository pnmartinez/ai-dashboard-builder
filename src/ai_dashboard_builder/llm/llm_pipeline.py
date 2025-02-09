"""Data pipeline for processing data through Language Learning Models (LLMs)."""

import hashlib
import json
import logging
import logging.handlers
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
import warnings

import numpy as np
import pandas as pd
import requests

from ..benchmarking import DashboardBenchmark
from ai_dashboard_builder.llm import prompts
from ai_dashboard_builder.utils.paths import get_root_path

# Configure logging
logger = logging.getLogger("LLMPipeline")
logger.setLevel(logging.WARNING)  # Set default to WARNING, we'll use INFO only for key metrics

# Add a file handler for debugging
debug_handler = logging.handlers.RotatingFileHandler(
    "llm_pipeline_debug.log",
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LLMPipeline:
    """A pipeline for processing data through various Language Learning Models (LLMs).

    This class handles interactions with different LLM providers (local and external),
    manages rate limiting, and provides methods for dataset analysis and visualization
    suggestions.

    Attributes:
        model_name (str): Name of the LLM model to use
        use_local (bool): Whether to use local (Ollama) or external API
        rate_limits (dict): Rate limiting configuration for different providers
        api_key (str): API key for external providers
        responses_dir (str): Directory to store LLM responses
    """

    def __init__(self, model_name: str = "mistral", use_local: bool = True):
        """Initialize the LLM pipeline.

        Args:
            model_name (str): Name of the LLM model to use
            use_local (bool): Whether to use local Ollama instance (True) or external API (False)

        Raises:
            ValueError: If API key is not found for external provider
        """
        self.model_name = model_name
        self.use_local = use_local
        
        # Get API key from environment based on model type
        if not use_local:
            if any(name in model_name.lower() for name in ["gpt", "o1"]):
                self.api_key = os.getenv("OPENAI_API_KEY")
                key_type = "OpenAI"
            elif "claude" in model_name.lower():
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                key_type = "Anthropic"
            elif "mistral" in model_name.lower():
                self.api_key = os.getenv("MISTRAL_API_KEY")
                key_type = "Mistral"
            elif any(name in model_name.lower() for name in ["mixtral", "groq", "llama", "gemma"]):
                self.api_key = os.getenv("GROQ_API_KEY")
                key_type = "Groq"
            else:
                self.api_key = os.getenv("LLM_API_KEY")
                key_type = "Generic"

            if not self.api_key:
                error_msg = f"No {key_type} API key found in environment for model: {model_name}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully initialized with {key_type} API key")

        # Rest of the initialization code...
        ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
        self.ollama_base_url = f"http://{ollama_host}:11434/api"

        # Add rate limiting configuration based on provider
        self.rate_limits = {
            "openai": 3,  # 3 seconds between calls for OpenAI
            "anthropic": 3,  # 3 seconds for Anthropic
            "mistral": 1,  # 1 second for Mistral
            "groq": 30,  # 30 seconds for Groq to respect TPM limits
            "default": 1,  # Default delay
        }

        # Track last API call time and token usage for Groq
        self.last_api_call = 0
        self.groq_tokens_used = 0
        self.groq_last_reset = time.time()

        # Update responses directory path to be relative to project root
        root_path = get_root_path()
        self.responses_dir = os.path.join(root_path, "src", "ai_dashboard_builder", "llm_responses")
        
        # Ensure the directory exists
        os.makedirs(self.responses_dir, exist_ok=True)
        logger.info(f"LLM responses directory: {self.responses_dir}")

        logger.info(
            f"Initializing LLMPipeline with model: {model_name} (local: {use_local})"
        )

    def _serialize_for_json(self, value: Any) -> Any:
        """Serialize a value for JSON encoding."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.float64, np.float32)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    def _time_execution(self, func_name: str, func, *args, **kwargs) -> Any:
        """Time the execution of a function and log the duration.

        Args:
            func_name (str): Name of the function being timed
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution
        """
        start_time = time.time()
        logger.info(f"Starting {func_name}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {func_name} in {duration:.2f} seconds")
        return result

    def _save_interaction(
        self, prompt: str, response: str, metadata: dict = None
    ) -> None:
        """Save LLM interaction details to disk for logging and debugging.

        Args:
            prompt (str): The prompt sent to the LLM
            response (str): The response received from the LLM
            metadata (dict, optional): Additional metadata about the interaction
        """
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
                "metadata": metadata or {},
            }

            # Save to file
            filepath = os.path.join(self.responses_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved interaction to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save interaction: {str(e)}", exc_info=True)

    def _get_rate_limit_delay(self) -> float:
        """Get the appropriate rate limit delay based on the model type.

        Returns:
            float: Number of seconds to wait between API calls
        """
        if any(name in self.model_name.lower() for name in ["gpt", "o1"]):
            return self.rate_limits["openai"]
        elif "claude" in self.model_name.lower():
            return self.rate_limits["anthropic"]
        elif "mistral" in self.model_name.lower():
            return self.rate_limits["mistral"]
        elif any(name in self.model_name.lower() for name in ["mixtral", "groq", "llama", "gemma"]):
            return self.rate_limits["groq"]
        return self.rate_limits["default"]

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls.

        Handles both standard rate limiting and special TPM (tokens per minute)
        limits for providers like Groq.
        """
        if not self.use_local:
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call
            delay_needed = self._get_rate_limit_delay()

            # Special handling for Groq's TPM limits
            if (
                "groq" in self.model_name
                or "mixtral" in self.model_name
                or "llama" in self.model_name
                or "gemma" in self.model_name
            ):
                # Reset token counter if a minute has passed
                if current_time - self.groq_last_reset >= 60:
                    self.groq_tokens_used = 0
                    self.groq_last_reset = current_time

                # If we're close to the TPM limit, wait for reset
                if self.groq_tokens_used >= 5000:  # Conservative limit (actual is 6000)
                    wait_time = 60 - (current_time - self.groq_last_reset)
                    if wait_time > 0:
                        logger.info(
                            f"Approaching Groq TPM limit, waiting {wait_time:.2f}s for reset"
                        )
                        time.sleep(wait_time)
                        self.groq_tokens_used = 0
                        self.groq_last_reset = time.time()

            # Standard rate limiting
            if time_since_last_call < delay_needed:
                sleep_time = delay_needed - time_since_last_call
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

            self.last_api_call = time.time()

    def _query_local(self, prompt: str) -> str:
        """Query local Ollama instance.

        Args:
            prompt (str): The prompt to send to the model

        Returns:
            str: Model's response or error message

        Raises:
            ValueError: If response format is unexpected
        """
        logger.info(f"Querying local Ollama model: {self.model_name}")
        try:
            start_time = time.time()
            logger.info("Sending request to Ollama API")
            response = requests.post(
                f"{self.ollama_base_url}/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )

            logger.info(
                f"Received response from Ollama (status: {response.status_code})"
            )

            if not response.ok:
                error_msg = f"Ollama API returned status code {response.status_code}"
                logger.error(error_msg)
                self._save_interaction(
                    prompt,
                    f"Error: {error_msg}",
                    {
                        "status": "error",
                        "error_type": "http_error",
                        "status_code": response.status_code,
                    },
                )
                return f"Error: {error_msg}"

            response_json = response.json()

            if "response" not in response_json:
                error_msg = f"Unexpected response format: {response_json}"
                logger.error(error_msg)
                self._save_interaction(
                    prompt,
                    f"Error: {error_msg}",
                    {
                        "status": "error",
                        "error_type": "format_error",
                        "raw_response": response_json,
                    },
                )
                raise ValueError(error_msg)

            response_text = response_json["response"]
            duration = time.time() - start_time

            # Save successful interaction
            self._save_interaction(
                prompt,
                response_text,
                {"status": "success", "duration_seconds": duration},
            )

            logger.info("Successfully processed response from local model")
            return response_text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error querying local model: {error_msg}", exc_info=True)
            self._save_interaction(
                prompt,
                f"Error: {error_msg}",
                {
                    "status": "error",
                    "error_type": "exception",
                    "error_message": error_msg,
                },
            )
            return f"Error: {error_msg}"

    def _query_api(self, prompt: str) -> str:
        """Query external API based on selected model."""
        self._enforce_rate_limit()

        start_time = time.time()
        try:
            response_text = ""

            if any(name in self.model_name.lower() for name in ["gpt", "o1"]):
                logger.info(f"Querying model: {self.model_name}")  # Keep model name logging
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                if any(name in self.model_name.lower() for name in ["o1-mini", "o1-preview"]):
                    data["max_completion_tokens"] = 4000
                    data["seed"] = 42
                else:
                    data["max_tokens"] = 2000
                    data["temperature"] = 0.7
                
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=60
                    )
                    
                    if response.status_code == 429:
                        error_data = response.json().get("error", {})
                        if error_data.get("code") == "insufficient_quota":
                            error_msg = "⚠️ API Quota Exceeded: Your OpenAI API key has exceeded its quota. Please check your billing details or try a different API key."
                        else:
                            error_msg = "⚠️ Rate Limit: Too many requests. Please try again in a few moments."
                        logger.error(error_msg)
                        return f"Error: {error_msg}"
                    elif response.status_code != 200:
                        error_msg = f"⚠️ OpenAI API Error ({response.status_code})"
                        logger.error(error_msg)
                        return f"Error: {error_msg}"
                    
                    response_json = response.json()
                    response_text = response_json["choices"][0]["message"]["content"]
                    
                    if not response_text.strip():
                        error_msg = "⚠️ Empty response from model. The input might be too long - try reducing the data size or using fewer columns."
                        logger.error(error_msg)
                        return f"Error: {error_msg}"
                    
                    logger.debug(f"Response received from {self.model_name}")  # Downgrade to debug
                    
                except requests.exceptions.Timeout:
                    error_msg = "⚠️ Request Timeout: The API request took too long to respond. Please try again."
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                except requests.exceptions.RequestException as e:
                    error_msg = f"⚠️ Network Error: Could not connect to OpenAI API. {str(e)}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                except Exception as e:
                    error_msg = f"⚠️ OpenAI API Error: {str(e)}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"

            elif "claude" in self.model_name:
                # Anthropic Claude API
                headers = {
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                }

                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000,
                    },
                )
                response_json = response.json()
                response_text = response_json["content"][0]["text"]

            elif "mistral" in self.model_name:
                # Mistral AI API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2000,
                    },
                )
                response_json = response.json()
                response_text = response_json["choices"][0]["message"]["content"]

            elif (
                "groq" in self.model_name
                or "mixtral" in self.model_name
                or "llama" in self.model_name
                or "gemma" in self.model_name
            ):
                # Groq API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                # Estimate token count (rough approximation)
                estimated_tokens = len(prompt.split()) * 1.3

                # Check if this would exceed TPM
                if self.groq_tokens_used + estimated_tokens >= 5000:
                    wait_time = 60 - (time.time() - self.groq_last_reset)
                    if wait_time > 0:
                        logger.info(f"Waiting {wait_time:.2f}s for Groq TPM reset")
                        time.sleep(wait_time)
                        self.groq_tokens_used = 0
                        self.groq_last_reset = time.time()

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2000,
                    },
                )
                response_json = response.json()

                # Update token usage if available in response
                if "usage" in response_json:
                    tokens_used = response_json["usage"].get(
                        "total_tokens", estimated_tokens
                    )
                    self.groq_tokens_used += tokens_used
                    logger.info(
                        f"Groq tokens used this minute: {self.groq_tokens_used}"
                    )
                else:
                    # Use estimate if not available
                    self.groq_tokens_used += estimated_tokens

                # Handle the response
                if "error" in response_json:
                    if "rate_limit_exceeded" in str(response_json["error"]):
                        # Wait and retry once if we hit rate limit
                        wait_time = 60
                        logger.info(
                            f"Hit Groq rate limit, waiting {wait_time}s before retry"
                        )
                        time.sleep(wait_time)
                        self.groq_tokens_used = 0
                        self.groq_last_reset = time.time()
                        return self._query_api(prompt)  # Retry the query
                    else:
                        raise ValueError(f"Groq API error: {response_json['error']}")

                response_text = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not response_text:
                    raise ValueError("Empty response from Groq API")

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            duration = time.time() - start_time
            logger.debug(f"Query completed in {duration:.2f}s")  # Downgrade to debug

            # Save successful interaction
            self._save_interaction(
                prompt,
                response_text,
                {
                    "status": "success",
                    "duration_seconds": duration,
                    "model": self.model_name,
                },
            )

            return response_text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error querying {self.model_name}: {error_msg}")
            return f"Error: {error_msg}"

    def _sort_dataframe_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe by temporal column if one exists."""
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    return df.sort_values(col)
            except Exception:
                continue
        return df

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """
        Check if a series contains datetime values.
        
        Args:
            series (pd.Series): The column to check
            
        Returns:
            bool: True if the column contains datetime values
        """
        try:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
            
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(series):
                return False
                
            # For object/string type columns, try parsing a sample
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                # Get a larger sample of non-null values for better accuracy
                sample = series.dropna().head(10)
                if len(sample) == 0:
                    return False
                
                success_count = 0
                total_count = len(sample)
                
                for val in sample:
                    if not isinstance(val, str):
                        continue
                        
                    try:
                        # Try dateutil parser first (more flexible)
                        from dateutil import parser
                        parser.parse(val)
                        success_count += 1
                    except (ImportError, ValueError, TypeError):
                        try:
                            # Fallback to pandas
                            pd.to_datetime(val, errors='raise')
                            success_count += 1
                        except (ValueError, TypeError):
                            continue
                
                # Consider it a datetime column if at least 80% of samples can be parsed
                return success_count / total_count >= 0.8
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking datetime column: {str(e)}")
            return False

    def analyze_dataset(self, df: pd.DataFrame, kpis: Optional[list] = None) -> str:
        """Perform comprehensive analysis of the dataset using LLM.

        Args:
            df (pd.DataFrame): Dataset to analyze
            kpis (list, optional): List of KPI columns to focus on
        """

        def _analyze(df: pd.DataFrame):
            logger.info("Starting dataset analysis")
            try:
                # Sort the dataframe chronologically if possible
                df = self._sort_dataframe_chronologically(df)

                # Create dataset summary
                data_summary = {
                    "columns": list(df.columns),
                    "sample_rows": [
                        {k: self._serialize_for_json(v) for k, v in row.items()}
                        for row in df.head(3).to_dict("records")
                    ],
                    "data_types": {
                        str(k): str(v) for k, v in df.dtypes.to_dict().items()
                    },
                    "null_counts": {
                        k: self._serialize_for_json(v)
                        for k, v in df.isnull().sum().to_dict().items()
                    },
                    "unique_counts": {
                        k: self._serialize_for_json(v)
                        for k, v in df.nunique().to_dict().items()
                    },
                }

                prompt = prompts.create_dataset_analysis_prompt(df, data_summary, kpis)

                # Query model and save response
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)

                return response

            except Exception as e:
                logger.error(f"Error in dataset analysis: {str(e)}", exc_info=True)
                raise

        return self._time_execution("analyze_dataset", _analyze, df)

    def suggest_visualizations(
        self,
        df: pd.DataFrame,
        kpis: Optional[List[str]] = None,
        filename: str = "unknown",
    ) -> Dict:
        """Generate visualization specifications for the dataset."""
        try:
            logger.debug("Starting visualization suggestions")
            
            # Get visualization suggestions from LLM
            viz_specs = self._get_visualization_suggestions(df, kpis)
            
            if isinstance(viz_specs, str) and viz_specs.startswith("Error:"):
                return viz_specs

            # Initialize benchmark system for scoring
            from ai_dashboard_builder.benchmarking.dashboard_benchmark import DashboardBenchmark
            
            # Use the warning suppression context manager
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="One or more sample arguments is too small")
                warnings.filterwarnings("ignore", message="all input arrays have length 1")
                
                benchmark = DashboardBenchmark(df)
                
                # Convert specs and get metrics
                benchmark_specs = []
                for spec in viz_specs.values():
                    benchmark_spec = spec.copy()
                    if isinstance(benchmark_spec.get('x'), list):
                        benchmark_spec['x'] = benchmark_spec['x'][0]
                    if isinstance(benchmark_spec.get('y'), list):
                        benchmark_spec['y'] = benchmark_spec['y'][0]
                    benchmark_specs.append(benchmark_spec)
                
                metrics = benchmark.evaluate_dashboard(benchmark_specs)
            
            # Log only the overall score
            logger.info(f"Model {self.model_name} - Overall Score: {metrics.overall_score():.3f}")
            
            return viz_specs

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in visualization suggestion: {error_msg}")
            return f"Error: {error_msg}"

    def explain_pattern(self, df: pd.DataFrame, pattern_description: str) -> str:
        """Get LLM explanation for a specific pattern in the data.

        Args:
            df (pd.DataFrame): The dataset
            pattern_description (str): Description of the pattern to analyze

        Returns:
            str: Pattern explanation
        """
        prompt = prompts.create_pattern_explanation_prompt(df, pattern_description)

        if self.use_local:
            return self._query_local(prompt)
        else:
            return self._query_api(prompt)

    def summarize_analysis(self, analysis: str, viz_specs: Dict[str, Any]) -> str:
        """Create a final summary of the analysis and visualizations.

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
                prompt = prompts.create_analysis_summary_prompt(analysis, viz_specs)

                # Query model
                logger.info("Querying model for summary")
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)

                logger.info("Successfully generated analysis summary")
                return response

            except Exception as e:
                logger.error(
                    f"Error in analysis summarization: {str(e)}", exc_info=True
                )
                return "Error generating summary: " + str(e)

        return self._time_execution("summarize_analysis", _summarize)

    def _get_visualization_suggestions(self, df: pd.DataFrame, kpis: Optional[List[str]] = None) -> Dict:
        """Generate visualization suggestions for the dataset using LLM."""
        logger.info("Starting visualization suggestions process")
        try:
            # Sort the dataframe chronologically if possible
            df = self._sort_dataframe_chronologically(df)

            # Initialize benchmark system for relationship analysis
            from ai_dashboard_builder.benchmarking.dashboard_benchmark import DashboardBenchmark
            benchmark = DashboardBenchmark(df)

            # Get column metadata with improved datetime detection
            logger.info("Creating column metadata")
            column_metadata = {
                col: {
                    "dtype": str(df[col].dtype),
                    "unique_count": self._serialize_for_json(df[col].nunique()),
                    "null_count": self._serialize_for_json(df[col].isnull().sum()),
                    "is_temporal": self._is_datetime_column(df[col]),
                    "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                    "is_categorical": pd.api.types.is_categorical_dtype(df[col])
                    or (df[col].nunique() / len(df) < 0.05),
                    "sample_values": [
                        self._serialize_for_json(v) for v in df[col].head().tolist()
                    ],
                }
                for col in df.columns
            }

            logger.info("Column metadata created successfully")

            # Generate the prompt
            sample_data = df.head(3).to_string()
            prompt = prompts.create_visualization_prompt(column_metadata, sample_data, kpis)
            logger.info("Generated visualization prompt")

            # Query model and save response
            if self.use_local:
                response = self._query_local(prompt)
            else:
                response = self._query_api(prompt)

            logger.info("Received visualization response")

            # If response starts with "Error:", return it directly
            if isinstance(response, str) and response.startswith("Error:"):
                return response

            # Extract and clean JSON from the response
            logger.info("Extracting JSON from response")
            viz_specs = self._extract_json_from_response(response)

            # Validate the visualization specifications
            if not isinstance(viz_specs, dict):
                logger.error("Parsed result is not a dictionary")
                return "Error: Failed to generate valid visualization specifications"

            # Add metadata to each visualization
            logger.info("Adding metadata to visualization specifications")
            validated_specs = {}

            for viz_id, viz_spec in viz_specs.items():
                # Ensure required fields are present
                required_fields = ["type", "x", "y", "color", "title", "description"]
                cleaned_spec = {}
                
                for field in required_fields:
                    cleaned_spec[field] = viz_spec.get(field, "")
                
                # Initialize parameters with defaults if not present
                cleaned_spec["parameters"] = viz_spec.get("parameters", {})
                if not isinstance(cleaned_spec["parameters"], dict):
                    cleaned_spec["parameters"] = {}
                
                # Process description and generate insight
                desc = cleaned_spec.get("description", "").strip()
                if desc:
                    # Extract quantitative information if present
                    numbers = re.findall(r'\d+(?:\.\d+)?', desc)
                    comparisons = re.findall(r'(more|less|higher|lower|increase|decrease|compared)', desc)
                    
                    insight = desc
                    if numbers or comparisons:
                        # Add actionable context if not present
                        if not any(term in desc for term in ["should", "could", "recommend", "suggest", "consider", "may want to"]):
                            insight = f"Based on this visualization, you may want to consider that {desc}"
                    cleaned_spec["insight"] = insight
                else:
                    # Generate basic insight based on chart type and columns
                    chart_type = cleaned_spec["type"]
                    x_col = cleaned_spec["x"]
                    y_col = cleaned_spec["y"]
                    
                    if chart_type == "scatter":
                        cleaned_spec["insight"] = f"This scatter plot reveals the relationship between {x_col} and {y_col}, which could help identify patterns or correlations."
                    elif chart_type == "bar":
                        cleaned_spec["insight"] = f"This bar chart shows the distribution of {y_col} across different {x_col} categories, highlighting key differences."
                    elif chart_type == "line":
                        cleaned_spec["insight"] = f"This line chart tracks changes in {y_col} over {x_col}, helping identify trends and patterns over time."
                    elif chart_type == "histogram":
                        cleaned_spec["insight"] = f"This histogram shows the distribution of {x_col}, which can help identify common values and outliers."
                    elif chart_type == "pie":
                        cleaned_spec["insight"] = f"This pie chart breaks down the composition of {x_col}, showing the relative proportions of each category."
                    else:
                        cleaned_spec["insight"] = f"This visualization shows the relationship between {x_col} and {y_col}."

                # Generate relationship text between x and y variables
                relationship_text = ""
                relationship_significance = False
                
                x = cleaned_spec.get('x')
                y = cleaned_spec.get('y')
                
                if x and y and x != y:  # Only proceed if both X and Y are present and different
                    rel = benchmark._find_relationship(x, y)
                    if rel:
                        p_value = rel.get('p_value')
                        stat = rel.get('stat')
                        rel_type = rel.get('type')
                        
                        if p_value is not None:
                            relationship_significance = p_value < 0.05
                            
                            if rel_type == 'numeric-numeric':
                                relationship_text = f"Relationship: {x} and {y} show a correlation of {stat:.3f} (p={p_value:.3f})"
                            elif rel_type == 'cat-cat':
                                relationship_text = f"Association: Chi-square test between {x} and {y} shows χ²={stat:.3f} (p={p_value:.3f})"
                            elif rel_type == 'numeric-cat':
                                num_var = x if x in benchmark.numeric_cols else y
                                cat_var = y if x in benchmark.numeric_cols else x
                                relationship_text = f"Group differences: ANOVA between {num_var} (numeric) and {cat_var} (categorical) shows F={stat:.3f} (p={p_value:.3f})"

                cleaned_spec["relationship_text"] = relationship_text
                cleaned_spec["relationship_significance"] = relationship_significance

                # Replace "None" color with default color
                if cleaned_spec.get("color") is None or (isinstance(cleaned_spec.get("color"), str) and cleaned_spec["color"].lower() in ["none", "null", "", "nan"]):
                    cleaned_spec["color"] = "#636EFA"  # Default Plotly blue

                # Handle metadata for x and y values that might be lists
                def get_column_metadata(col_name):
                    if isinstance(col_name, list):
                        return [column_metadata.get(c, {}) for c in col_name]
                    return column_metadata.get(col_name, {})

                # Add metadata
                cleaned_spec["metadata"] = {
                    "x_meta": get_column_metadata(cleaned_spec["x"]),
                    "y_meta": get_column_metadata(cleaned_spec["y"]),
                    "color_meta": get_column_metadata(cleaned_spec.get("color")) if cleaned_spec.get("color") else {},
                }

                # Add default parameters
                cleaned_spec["parameters"].update(
                    {"height": 400, "template": "plotly", "opacity": 0.8}
                )

                validated_specs[viz_id] = cleaned_spec

            logger.info(f"Successfully validated {len(validated_specs)} visualization specifications")
            return validated_specs

        except Exception as e:
            logger.error(f"Error in visualization suggestion: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return {}

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from the LLM response, with robust error handling.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            Dict[str, Any]: The parsed JSON object or a default visualization if parsing fails
        """
        try:
            # If response is already a dict, return it
            if isinstance(response, dict):
                return response

            # First try to find JSON between triple backticks
            json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no backticks found, try to find a JSON object directly
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.error("No JSON object found in response")
                    return self._create_default_visualization()

            # Common JSON fixes
            json_str = json_str.replace("None", "null")  # Replace Python None with JSON null
            json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)  # Remove trailing commas
            json_str = re.sub(r"(\w+):", r'"\1":', json_str)  # Quote unquoted keys
            json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)  # Fix missing commas between fields
            json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)  # Fix missing commas between objects
            
            # Fix truncated JSON by completing missing structures
            open_braces = json_str.count("{")
            close_braces = json_str.count("}")
            if open_braces > close_braces:
                json_str += "}" * (open_braces - close_braces)
            
            # Try to parse the fixed JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {str(e)}")
                
                # Additional fixes for specific issues
                if "Expecting value" in str(e):
                    json_str = re.sub(r":\s*,", ": null,", json_str)
                    json_str = re.sub(r":\s*}", ": null}", json_str)
                
                # Try to extract any complete visualization specs
                try:
                    # Find all complete viz specs
                    viz_specs = {}
                    viz_pattern = r'"viz_\d+":\s*\{[^}]*\}'
                    matches = re.finditer(viz_pattern, json_str)
                    
                    for match in matches:
                        try:
                            viz_json = "{" + match.group(0) + "}"
                            viz_data = json.loads(viz_json)
                            viz_specs.update(viz_data)
                        except json.JSONDecodeError:
                            continue
                    
                    if viz_specs:
                        logger.info(f"Recovered {len(viz_specs)} complete visualization specs")
                        return viz_specs
                    
                except Exception as viz_error:
                    logger.error(f"Failed to extract partial visualizations: {str(viz_error)}")

                # If all else fails, return a default visualization
                logger.warning("Falling back to default visualization")
                return self._create_default_visualization()

        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return self._create_default_visualization()

    def _create_default_visualization(self) -> Dict[str, Any]:
        """Create a default visualization when parsing fails."""
        return {
            "viz_1": {
                "type": "bar",
                "x": self.df.columns[0] if hasattr(self, 'df') and len(self.df.columns) > 0 else "column",
                "y": "count",
                "color": None,
                "title": "Data Overview",
                "description": "Basic overview of the dataset.",
                "parameters": {
                    "orientation": "v",
                    "aggregation": "count"
                }
            }
        }


# Example usage:
if __name__ == "__main__":
    # Example with local Ollama
    pipeline = LLMPipeline(model_name="mistral", use_local=True)

    # Load your dataset
    df = pd.read_csv("your_dataset.csv")  # Replace with your data loading

    # Get analysis
    analysis = pipeline.analyze_dataset(df)
    print(analysis)
