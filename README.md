# AI Dashboard Builder

Drop your data and let the AI make a dashboard and extract insights.

This is currently structured as a simple Plotly app that dynamically builds a dashboard by analyzing the dataset with LLMs. You can:
- Use local LLMs (ollama)
- Use external providers by providing your API KEY

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed on your system
- (Optional) API keys for external LLM providers

### Running with Local LLM (Ollama)

1. Clone the repository:
```bash
git clone https://github.com/pnmartinez/ai-dashboard-builder.git
```

2. Run the Docker container:
```bash
docker compose up --build
```

3. Open your browser and navigate to `http://localhost:8050` to see the dashboard.
