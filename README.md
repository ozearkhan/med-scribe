# Psychiatry Section Classifier

> Embedding-based classification system for psychiatric clinical notes with LLM reranking.

## Features

- **Section Classification** - Identify note sections (CC, HPI, MSE, Assessment, etc.)
- **Safety Screening** - Detect suicidal/homicidal ideation  
- **Domain Classification** - Classify symptom domains (mood, anxiety, cognition)
- **LLM Reranking** - Optional Gemini-powered result refinement
- **Section Splitting** - Process notes section-by-section for better accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)            │
│  Enhanced Pipeline → Section View → Results Display         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP (port 8001)
┌─────────────────────▼───────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  /process/note/enhanced → Section Splitter → Classifier     │
├─────────────────────────────────────────────────────────────┤
│  Services:                                                  │
│  ├── EmbeddingClassifier (sentence-transformers)            │
│  ├── SectionSplitter (keyword + rule-based)                 │
│  ├── GeminiClient (optional LLM reranking)                  │
│  └── EnhancedPipelineProcessor                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Backend Setup

```bash
cd ui/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure (optional - for LLM reranking)
copy .env.example .env
# Edit .env and add GEMINI_API_KEY

# Start server
python -m uvicorn backend_api:app --reload --port 8001
```

### 2. Frontend Setup

```bash
cd ui/frontend

# Install dependencies
npm install

# Start dev server
npm start
```

Open http://localhost:3000

## Datasets

| Dataset | Classes | Purpose |
|---------|---------|---------|
| `psychiatry_sections` | 9 | Section type (CC, HPI, MSE, etc.) |
| `psychiatry_domains` | 8 | Symptom domains (mood, anxiety, etc.) |
| `psychiatry_safety` | 4 | SI/HI screening |

## Project Structure

```
rag_scribe/
├── docs/                    # Research & architecture docs
├── scripts/                 # Utility scripts
│   ├── evaluate_classifier.py
│   └── generate_embeddings.py
└── ui/                      # Main application
    ├── backend/
    │   ├── backend_api.py   # FastAPI server
    │   ├── services/
    │   │   ├── embedding_classifier.py
    │   │   ├── enhanced_pipeline.py
    │   │   ├── section_splitter.py
    │   │   └── gemini_client.py
    │   └── requirements.txt
    ├── frontend/            # React UI
    └── data/
        ├── datasets/        # JSON class definitions
        └── embeddings/      # Pre-computed embeddings
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/datasets` | GET | List available datasets |
| `/classify/text` | POST | Single classifier |
| `/process/note` | POST | Full pipeline (all classifiers) |
| `/process/note/enhanced` | POST | Section-splitting pipeline |

## Configuration

Edit `ui/backend/.env`:

```ini
GEMINI_API_KEY=your-api-key-here  # Optional, for LLM reranking
GEMINI_DEFAULT_MODEL=gemini-2.0-flash-exp
```

## Documentation

See `docs/` for detailed documentation:
- `AI_SCRIBE_ARCHITECTURE.md` - Full system architecture
- `RESEARCH_SYNTHESIS.md` - Research paper analysis
- `TECHNICAL_KT_DOCUMENTATION.md` - Technical deep-dive

## License

MIT
