# AI Dashboard Builder 📊🤖 
![CI Status](https://github.com/pnmartinez/ai-dashboard-builder/actions/workflows/docker-compose-all-in-one-test.yml/badge.svg)

### Features

* **Privacy** 🛡️: you can use local LLMs through Ollama for your private data.
* **Flexibility** 🤸: make the LLMs focus on your priorities by selecting your KPIs in the dataset, or limit the dataset by columns and/or rows.
* **Reusability** 🔄: each dashboard generates a reusable "viz_spec" JSON file in the "llm_responses" folder, accessible for future use through the "Import Previous Viz Specs" feature.


## Setup

### Option 1: Docker Compose (Recommended)

1. Start the application with Docker Compose:
```bash
docker-compose up --build
```

2. Access the dashboard at http://localhost:8050

### Option 2: All-in-One Deployment (includes Ollama)

1. Start both Ollama and the dashboard:
```bash
docker-compose -f docker-compose.all-in-one.yml up --build
```

2. Access the dashboard at http://localhost:8050

### Option 3: Manual Deployment

1. Start Ollama separately (if using local models)
2. Run the application:
```bash
python src/app.py
```

## Development

To run the application in development mode:
```bash
PYTHONPATH=$PYTHONPATH:./src python src/app.py
```

## Setting API KEYS and/or Olama

In the project root folder, you can create a `.env` file and set the API keys for the LLMs you want to use, or pass them through the webapp.

- `OLLAMA_HOST`: Ollama server address (default: host.docker.internal)
- `OPENAI_API_KEY`: OpenAI API key (for GPT models)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude models)
- `GROQ_API_KEY`: Groq API key (for Mixtral/LLaMA models)


## License

This project is licensed under a form of MIT License - see the [LICENSE.md](LICENSE.md) file for details.
