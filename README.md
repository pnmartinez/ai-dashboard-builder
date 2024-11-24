# AI Dashboard Builder

## Project Structure
```
project_root/
├── src/               # Source code
│   ├── app.py        # Main application
│   ├── dashboard_builder.py
│   └── llm/          # LLM pipeline module
├── docker/           # Docker configuration
├── requirements.txt
└── README.md
```

## Development Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.default .env
# Edit .env with your API keys
```

## Deployment

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

## Environment Variables

- `OLLAMA_HOST`: Ollama server address (default: host.docker.internal)
- `OPENAI_API_KEY`: OpenAI API key (for GPT models)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude models)
- `GROQ_API_KEY`: Groq API key (for Mixtral/LLaMA models)

## Development

To run the application in development mode:
```bash
PYTHONPATH=$PYTHONPATH:./src python src/app.py
```
