# Psychiatry AI Scribe - UI Demo
## Classification System for Psychiatric Clinical Notes

This UI provides a visual interface for classifying psychiatric clinical notes using:
- **Embedding Similarity** (sentence-transformers, runs locally)
- **LLM Reranking** (Gemini API)
- **Attribute Validation** (Gemini API)

## Quick Start

### 1. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Setup Frontend

```bash
cd frontend
npm install
```

### 3. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn backend_api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Open http://localhost:3000 in your browser.

## Configuration

Edit `backend/.env`:

```ini
GEMINI_API_KEY=your-api-key-here
GEMINI_DEFAULT_MODEL=gemini-2.0-flash-exp
```

## Available Datasets

The UI pre-loads these psychiatry classification datasets:
- **Section Classification** - SOAP sections, safety screening
- **Domain Classification** - Mood, anxiety, cognition, etc.
- **Entity Classification** - Medications, symptoms, procedures
- **Relation Classification** - Clinical relationships
- **Safety Classification** - SI/HI screening

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  Classification Page → Options → Results Display            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  /classify/text → EmbeddingClassifier → Response            │
├─────────────────────────────────────────────────────────────┤
│  Embedding Classifier                                       │
│  ├── sentence-transformers (local)                          │
│  ├── Gemini Reranking (API)                                │
│  └── Gemini Attribute Validation (API)                     │
└─────────────────────────────────────────────────────────────┘
```