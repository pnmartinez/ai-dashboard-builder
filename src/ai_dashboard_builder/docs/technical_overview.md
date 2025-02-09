# AI Dashboard Builder - Technical Overview

## Architecture Overview

The AI Dashboard Builder is a Plotly Dash application that leverages Language Learning Models (LLMs) to automatically generate data visualizations. The application follows a modular architecture with these key components:

### Core Components

1. **LLM Pipeline** (`llm_pipeline.py`)
   - Handles interactions with LLM providers (local Ollama or external APIs)
   - Manages rate limiting and token usage
   - Processes dataset analysis and visualization suggestions
   - Caches LLM responses for debugging

2. **Dashboard Builder** (`dashboard_builder.py`)
   - Converts LLM visualization specifications into Plotly figures
   - Handles figure styling and layout
   - Generates reproducible code for each visualization

3. **Dashboard Benchmark** (`dashboard_benchmark.py`)
   - Evaluates scientific validity of visualizations
   - Analyzes relationships between variables
   - Scores visualizations on multiple metrics:
     * Validity (0-1): Technical correctness
     * Relevance (0-1): Justification clarity
     * Usefulness (0-1): Actionable insights
     * Diversity (0-1): Visualization variety
     * Redundancy (0-1): Penalty for duplicate insights

## Data Flow

1. **Data Upload & Processing**
   ```
   Upload → Smart Type Conversion → Preview Generation → Data Store
   ```
   - Handles CSV and Excel files
   - Performs intelligent numeric conversion
   - Stores data in memory using Dash's dcc.Store

2. **LLM Processing Pipeline**
   ```
   Data → Analysis → Viz Suggestions → Benchmark → Dashboard
   ```
   - Dataset analysis (optional)
   - Visualization specification generation
   - Quality benchmarking
   - Dashboard rendering

3. **Filter System**
   ```
   Raw Data → Dynamic Filter Generation → Filter State → Filtered Visualizations
   ```
   - Automatically infers temporal and categorical filters
   - Updates visualizations reactively

## Key Callbacks

### Critical Callbacks

1. **Data Upload** (`handle_upload`)
   - Triggers: `upload-data.contents`
   - Purpose: Process uploaded file and prepare data preview
   - Critical considerations: 
     * Handles large files through preview limits
     * Performs smart type conversion
     * Maintains original data in store

2. **Analysis Pipeline** (`analyze_data`)
   - Triggers: `analyze-button.n_clicks`
   - Purpose: Core analysis and visualization generation
   - Uses long_callback to prevent timeout
   - Progress tracking through set_progress
   - Critical considerations:
     * Rate limiting between API calls
     * Error handling for API failures
     * Caching of responses

3. **Filter System** (`update_filter_state`, `update_visualizations`)
   - Triggers: Filter component changes
   - Purpose: Maintain filter state and update visualizations
   - Critical considerations:
     * Maintains filter state across components
     * Handles both temporal and categorical filters
     * Updates all visualizations efficiently

### State Management

The application uses multiple dcc.Store components:
- `data-store`: Raw dataset and metadata
- `viz-state`: Visualization generation state
- `dashboard-rendered`: Dashboard render status
- `filter-state`: Current filter configurations
- `benchmark-results`: Benchmark metrics and scores

## User Stories & Features

1. **Basic Usage**
   ```python
   # User uploads data
   # System automatically:
   - Previews data with configurable limits
   - Enables analysis options
   - Suggests KPI columns
   ```

2. **Analysis Options**
   ```python
   # User can:
   - Select LLM provider (local/external)
   - Configure API keys
   - Choose specific models
   - Select KPIs of interest
   - Toggle text insights
   ```

3. **Dashboard Interaction**
   ```python
   # Generated dashboard provides:
   - Dynamic filters (temporal/categorical)
   - Expandable visualizations
   - Code reproduction tabs
   - Relationship insights
   ```

4. **Benchmarking**
   ```python
   # Full benchmark system:
   - Tests multiple models
   - Multiple passes per model
   - Comprehensive scoring
   - Visual comparison of results
   ```

## Critical Considerations

1. **Rate Limiting**
   - Each LLM provider has specific rate limits
   - Token usage tracking for Groq
   - Automatic retry logic for rate limit errors

2. **Error Handling**
   - Graceful degradation for API failures
   - User-friendly error messages
   - Detailed logging for debugging

3. **Performance**
   - Preview limits for large datasets
   - Efficient filter updates
   - Caching of LLM responses

4. **State Management**
   - Multiple dcc.Store components
   - Careful handling of callback chains
   - Prevention of unnecessary updates

## Future Considerations

1. **Extensibility**
   - New visualization types can be added to DashboardBuilder
   - Additional benchmark metrics can be implemented
   - New LLM providers can be integrated

2. **Potential Issues**
   - Memory management for large datasets
   - API rate limiting during benchmarking
   - State synchronization in complex filter scenarios

This technical overview should be updated as the application evolves to maintain its usefulness as a reference. 