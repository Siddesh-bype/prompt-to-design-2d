# Prompt-to-Blueprint AI

Convert natural language descriptions into 2D architectural floor plans using AI.

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd prompt-to-blueprint
cp .env.example .env

# 2. Start all services
docker-compose up -d

# 3. Pull the Ollama model
docker exec -it prompt-to-blueprint-ollama-1 ollama pull qwen2.5:1.5b

# 4. Open the app
open http://localhost:3000
```

## Architecture

```
User Prompt → NLP Parser (Ollama SLM) → GNN Layout Engine → Z3 Constraint Solver → SVG/DXF Renderer
```

## Tech Stack

- **Backend**: Python 3.11, FastAPI, PyTorch Geometric, Z3 Solver, Shapely
- **Frontend**: React 18, Vite 5, TailwindCSS, Konva.js, Zustand
- **Infrastructure**: Docker, Redis + RQ, Ollama

## Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## API Endpoints

| Method | Endpoint              | Description            |
|--------|-----------------------|------------------------|
| POST   | /api/v1/generate      | Submit floor plan job  |
| GET    | /api/v1/jobs/{job_id} | Get job status/result  |
| WS     | /ws/{job_id}          | Real-time job updates  |
| POST   | /api/v1/vastu         | Check Vastu compliance |
| POST   | /api/v1/export/dxf    | Export as DXF          |
| POST   | /api/v1/export/svg    | Export as SVG          |
| GET    | /api/v1/health        | Health check           |
