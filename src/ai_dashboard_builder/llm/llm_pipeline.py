"""Data pipeline for processing data through Language Learning Models (LLMs)."""

import hashlib
import json
import logging
import logging.handlers
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

from ai_dashboard_builder.llm import prompts

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMPipeline")
logger.setLevel(logging.INFO)

# Add a file handler for debugging
debug_handler = logging.handlers.RotatingFileHandler(
    "llm_pipeline_debug.log",
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)


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

        # Update responses directory path to be relative to src folder
        src_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # Get src directory
        self.responses_dir = os.path.join(src_dir, "llm_responses")
        if not os.path.exists(self.responses_dir):
            os.makedirs(self.responses_dir)

        logger.info(
            f"Initializing LLMPipeline with model: {model_name} (local: {use_local})"
        )

        if not use_local:
            # Get API key based on model
            if "gpt" in model_name:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif "claude" in model_name:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif "mistral" in model_name:
                self.api_key = os.getenv("MISTRAL_API_KEY")
            elif (
                "mixtral" in model_name
                or "groq" in model_name
                or "llama" in model_name
                or "gemma" in model_name
            ):
                self.api_key = os.getenv("GROQ_API_KEY")
            else:
                self.api_key = os.getenv("LLM_API_KEY")

            if not self.api_key:
                logger.error("API key not found for model: " + model_name)
                raise ValueError(f"API key not found for model: {model_name}")

    def _serialize_for_json(self, obj) -> Any:
        """Convert various data types to JSON-serializable format.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable version of the input object
        """
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
        if "gpt" in self.model_name:
            return self.rate_limits["openai"]
        elif "claude" in self.model_name:
            return self.rate_limits["anthropic"]
        elif "mistral" in self.model_name:
            return self.rate_limits["mistral"]
        elif (
            "mixtral" in self.model_name
            or "groq" in self.model_name
            or "llama" in self.model_name
            or "gemma" in self.model_name
        ):
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
        """Query external API based on selected model.

        Handles different API providers (OpenAI, Anthropic, Mistral, Groq)
        with appropriate rate limiting and error handling.

        Args:
            prompt (str): The prompt to send to the model

        Returns:
            str: Model's response or error message

        Raises:
            ValueError: If model is unsupported or API returns error
        """
        # Add rate limiting before making the API call
        self._enforce_rate_limit()

        start_time = time.time()
        try:
            response_text = ""

            if "gpt" in self.model_name:
                # OpenAI GPT API (>=1.0.0)
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000,
                )
                response_text = response.choices[0].message.content

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
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(
                f"Error querying {self.model_name} API: {error_msg}", exc_info=True
            )

            # Save failed interaction
            self._save_interaction(
                prompt,
                f"Error: {error_msg}",
                {
                    "status": "error",
                    "error_type": "api_error",
                    "error_message": error_msg,
                    "duration_seconds": duration,
                    "model": self.model_name,
                },
            )

            return f"Error: {error_msg}"

    def _sort_dataframe_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to sort the dataframe chronologically by detecting date/time columns.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Sorted dataframe if temporal column found, otherwise original df
        """
        try:
            # List of common date/time column names (case-insensitive)
            date_column_patterns = [
                r"date",
                r"time",
                r"timestamp",
                r"datetime",
                r"created.*at",
                r"updated.*at",
                r"modified.*at",
                r"year",
                r"month",
                r"day",
                r"period",
            ]

            # Find columns that match date patterns
            potential_date_cols = []
            for col in df.columns:
                # Check if column name matches any date pattern
                if any(
                    re.search(pattern, col.lower()) for pattern in date_column_patterns
                ):
                    potential_date_cols.append(col)

            # Also check for columns that look like dates
            for col in df.columns:
                if col not in potential_date_cols:
                    # Sample some values to check if they're dates
                    sample = df[col].dropna().head()
                    try:
                        pd.to_datetime(sample, errors="raise")
                        potential_date_cols.append(col)
                    except (ValueError, TypeError):
                        continue

            if potential_date_cols:
                # Try each potential date column
                for col in potential_date_cols:
                    try:
                        # Convert to datetime and sort
                        df[col] = pd.to_datetime(df[col], errors="raise")
                        sorted_df = df.sort_values(by=col)
                        logger.info(f"Successfully sorted dataframe by {col}")
                        return sorted_df
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not sort by {col}: {str(e)}")
                        continue

            logger.info("No suitable datetime column found for sorting")
            return df

        except Exception as e:
            logger.warning(f"Error while attempting to sort dataframe: {str(e)}")
            return df

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
        kpis: Optional[list] = None,
        filename: str = "unknown_file",
    ) -> Dict[str, Any]:
        """Generate visualization suggestions for the dataset using LLM.

        Args:
            df (pd.DataFrame): Dataset to visualize
            kpis (list, optional): List of KPI columns to focus on
            filename (str): Name of the original data file
        """

        def _suggest(df: pd.DataFrame):
            logger.info("Starting visualization suggestions process")
            try:
                # Sort the dataframe chronologically if possible
                df = self._sort_dataframe_chronologically(df)

                # Get column metadata
                logger.info("Creating column metadata")
                column_metadata = {
                    col: {
                        "dtype": str(df[col].dtype),
                        "unique_count": self._serialize_for_json(df[col].nunique()),
                        "null_count": self._serialize_for_json(df[col].isnull().sum()),
                        "is_temporal": pd.api.types.is_datetime64_any_dtype(df[col]),
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

                prompt = prompts.create_visualization_prompt(
                    column_metadata, sample_data, kpis
                )

                logger.info("Generated visualization prompt")

                # Query model and save response
                if self.use_local:
                    response = self._query_local(prompt)
                else:
                    response = self._query_api(prompt)

                logger.info("Received visualization response")

                # Extract and clean JSON from the response
                logger.info("Extracting JSON from response")

                # Find JSON structure with better error handling
                def extract_json_from_response(self, response: str) -> Dict[str, Any]:
                    """Extract and parse JSON from the LLM response, with better error handling and JSON fixing."""
                    try:
                        # If response is already a dict, return it
                        if isinstance(response, dict):
                            return response

                        # First try to find JSON between triple backticks
                        json_match = re.search(
                            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response
                        )
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # If no backticks found, try to find a JSON object directly
                            json_match = re.search(r"\{[\s\S]*\}", response)
                            if json_match:
                                json_str = json_match.group(0)
                            else:
                                raise ValueError("No JSON object found in response")

                        # Common JSON fixes
                        json_str = json_str.replace(
                            "None", "null"
                        )  # Replace Python None with JSON null
                        json_str = re.sub(
                            r",(\s*[}\]])", r"\1", json_str
                        )  # Remove trailing commas
                        json_str = re.sub(
                            r"(\w+):", r'"\1":', json_str
                        )  # Quote unquoted keys

                        # Fix missing commas between fields
                        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
                        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)

                        # Try to parse the fixed JSON
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Initial JSON parsing failed: {str(e)}")

                            # Additional fixes for specific issues
                            if "Expecting value" in str(e):
                                # Try to fix missing values
                                json_str = re.sub(r":\s*,", ": null,", json_str)
                                json_str = re.sub(r":\s*}", ": null}", json_str)

                            # Try parsing again after additional fixes
                            return json.loads(json_str)

                    except Exception as e:
                        logger.error(f"JSON parsing error: {str(e)}")

                        # Last resort: try ast.literal_eval
                        try:
                            logger.info("Attempting to parse with ast.literal_eval")
                            import ast

                            # Replace null with None for Python parsing
                            if isinstance(response, str):
                                response = response.replace("null", "None")
                            parsed = ast.literal_eval(str(response))
                            return parsed
                        except Exception as e2:
                            logger.error(
                                f"Failed to parse JSON even with ast.literal_eval: {str(e2)}"
                            )
                            raise ValueError(
                                f"Could not parse response as JSON: {str(e)}"
                            )

                def clean_json_str(json_str: Any) -> Any:
                    """Clean JSON string or return the input if it's already parsed."""
                    # If input is already a dict, return it
                    if isinstance(json_str, dict):
                        return json_str

                    # If input is a string, clean it
                    if isinstance(json_str, str):
                        # Remove any markdown formatting
                        json_str = json_str.replace("```json", "").replace("```", "")

                        # Remove any control characters
                        json_str = "".join(
                            char
                            for char in json_str
                            if ord(char) >= 32 or char in "\n\r\t"
                        )

                        # Fix common JSON formatting issues
                        json_str = json_str.replace("\n", " ").replace("\r", " ")
                        json_str = json_str.replace("\\n", " ").replace("\\r", " ")
                        json_str = json_str.strip()

                    return json_str

                try:
                    # If response is already a dict, use it directly
                    if isinstance(response, dict):
                        viz_specs = response
                    else:
                        # Extract JSON from string response
                        json_str = extract_json_from_response(self, response)
                        if isinstance(json_str, dict):
                            viz_specs = json_str
                        else:
                            json_str = clean_json_str(json_str)
                            viz_specs = json.loads(json_str)

                    logger.info(
                        f"Successfully parsed JSON with {len(viz_specs)} visualization specifications"
                    )

                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing error: {str(je)}")
                    logger.debug(f"Problematic JSON string: {response}")

                    # Attempt to fix common JSON issues
                    try:
                        # Try eval as a last resort (safe for dict literals)
                        import ast

                        viz_specs = ast.literal_eval(str(response))
                        logger.info("Successfully parsed JSON using ast.literal_eval")
                    except Exception:
                        logger.error("Failed to parse JSON even with ast.literal_eval")
                        return {}

                # Validate the visualization specifications
                if not isinstance(viz_specs, dict):
                    logger.error("Parsed result is not a dictionary")
                    return {}

                # Add metadata to each visualization
                logger.info("Adding metadata to visualization specifications")
                validated_specs = {}

                for viz_id, viz_spec in viz_specs.items():
                    # Ensure required fields are present
                    if not all(key in viz_spec for key in ["type", "title"]):
                        logger.warning(
                            f"Skipping invalid visualization spec {viz_id}: missing required fields"
                        )
                        continue

                    # Clean and validate the specification
                    cleaned_spec = {
                        "type": str(viz_spec.get("type", "")).lower(),
                        "x": str(viz_spec.get("x", "")),
                        "y": str(viz_spec.get("y", "")),
                        "color": str(viz_spec.get("color", "")),
                        "title": str(viz_spec.get("title", "")),
                        "description": str(viz_spec.get("description", "")),
                        "parameters": viz_spec.get("parameters", {}),
                    }

                    # Replace "None" color with default color
                    if cleaned_spec["color"].lower() in ["none", "null", "", None]:
                        cleaned_spec["color"] = "#636EFA"  # Default Plotly blue

                    # Add metadata
                    cleaned_spec["metadata"] = {
                        "x_meta": column_metadata.get(cleaned_spec["x"], {}),
                        "y_meta": column_metadata.get(cleaned_spec["y"], {}),
                        "color_meta": column_metadata.get(cleaned_spec["color"], {}),
                    }

                    # Add default parameters
                    cleaned_spec["parameters"].update(
                        {"height": 400, "template": "plotly", "opacity": 0.8}
                    )

                    validated_specs[viz_id] = cleaned_spec

                # Save visualization specifications with model name and filename in the metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_model_name = re.sub(r"[^\w\-]", "_", self.model_name)
                viz_specs_file = os.path.join(
                    self.responses_dir, f"viz_specs_{clean_model_name}_{timestamp}.json"
                )

                try:
                    with open(viz_specs_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "timestamp": timestamp,
                                "model": self.model_name,
                                "provider": "local" if self.use_local else "api",
                                "dataset_filename": filename,  # Add filename to metadata
                                "column_metadata": column_metadata,
                                "visualization_specs": validated_specs,
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(
                        f"Saved visualization specifications to {viz_specs_file}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save visualization specifications: {str(e)}",
                        exc_info=True,
                    )

                return validated_specs

            except Exception as e:
                logger.error(
                    f"Error in visualization suggestion: {str(e)}", exc_info=True
                )
                return {}

        return self._time_execution("suggest_visualizations", _suggest, df)

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


# Example usage:
if __name__ == "__main__":
    # Example with local Ollama
    pipeline = LLMPipeline(model_name="mistral", use_local=True)

    # Load your dataset
    df = pd.read_csv("your_dataset.csv")  # Replace with your data loading

    # Get analysis
    analysis = pipeline.analyze_dataset(df)
    print(analysis)
