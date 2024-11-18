# AI Dashboard Builder

Drop your data and let the AI make a dashboard and extract insights.

This is currently structured as a simple Plotly app that dynamically builds a dashboard by analyzing the dataset with LLMs. You can:
- Use local LLMs (ollama)
- Use external providers by providing your API KEY.

![image](https://github.com/user-attachments/assets/5f266f41-34c8-4d2c-b689-40ace28433da)

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed on your system
- (Optional) API keys for external LLM providers

### API Key Setup (Optional)
If you want to use external LLM providers, you can supply the KEY in the UI or create a `.env` file in the root directory with your API keys:

```bash
# .env file
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

### Option 1: Running with External Ollama

1. Clone the repository:
```bash
git clone https://github.com/pnmartinez/ai-dashboard-builder.git
```

2. Ensure ollama has llama3.1 model pulled. If not, run `ollama pull llama3.1`.

3. Run the Docker container:
```bash
docker compose up --build
```

### Option 2: Running with Bundled Ollama

1. Clone the repository:
```bash
git clone https://github.com/pnmartinez/ai-dashboard-builder.git
```

2. Go into the folder and run the app using the all-in-one compose file:
```bash
docker compose -f docker-compose.all-in-one.yml up --build
```

The bundled version will automatically pull and set up ollama with the required model (llama3.1). First run could take some minutes to pull the model.

3. For either option, open your browser and navigate to `http://localhost:8050` to see the dashboard.
