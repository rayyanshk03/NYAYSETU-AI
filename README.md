# NyayaSetu AI — Multilingual Agentic AI-Based Family Law Information and Decision-Support System

> **Step 1: Foundation (Frontend & Backend Integration) — COMPLETED**  
> **Step 2: Legal Document Ingestion & Knowledge Base Preparation — COMPLETED**  
> **Step 3: Retrieval-Augmented Generation (RAG) & Vector Database — COMPLETED**  
> **Step 4: Agentic AI Layer (Planner, Synthesizer, Orchestrator & UI Trace) — COMPLETED**

NyayaSetu AI is an agentic AI-driven legal information and decision-support platform designed specifically for Indian family law across **English**, **Hindi**, and **Kannada**.

---

## Architecture Overview

```
nyayasetu-ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application, CORS middleware, route mounting
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── endpoints.py     # GET /health, POST /chat, POST /retrieve, GET /corpus/status
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py        # Pydantic BaseSettings, strict CORS origin controls
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── health.py        # HealthResponse schema
│   │   │   ├── chat.py          # ChatRequest, ChatResponse, SupportedLanguage enum
│   │   │   └── retrieval.py     # [STEP 3] RetrievalRequest, RetrievalResponse, RetrievedPassage
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── chat_service.py  # ChatService bridge connecting /api/chat to LegalAgent
│   │   ├── ingestion/           # [STEP 2] Ingestion & Knowledge Base Pipeline
│   │   │   ├── __init__.py      # Ingestion package exports
│   │   │   ├── extract.py       # PyMuPDF text extractor & scanned PDF detector
│   │   │   ├── clean.py         # Conservative text cleaner & header/footer remover
│   │   │   ├── chunk.py         # Legal section- & subsection-aware chunker
│   │   │   ├── metadata.py      # Source manifest manager & India Code registry
│   │   │   └── pipeline.py      # CLI orchestrator & batch processor
│   │   ├── rag/                 # [STEP 3] Vector Store & Multilingual Retrieval
│   │   │   ├── __init__.py      # RAG package exports
│   │   │   ├── embeddings.py    # Multilingual SentenceTransformer embedder with mock fallback
│   │   │   ├── vector_store.py  # ChromaDB persistent collection manager
│   │   │   ├── retrieval.py     # LegalRetriever with similarity thresholding & deduplication
│   │   │   ├── indexer.py       # Batch indexing CLI for processed JSONL chunks
│   │   │   └── evaluator.py     # Retrieval benchmark & precision/recall evaluation
│   │   └── agent/               # [STEP 4] Agentic AI Planning, Execution & Synthesis
│   │       ├── __init__.py      # Agent package exports
│   │       ├── planner.py       # QueryPlanner: intent classifier & sub-query generator
│   │       ├── synthesizer.py   # LegalSynthesizer: citation-backed answers & safety disclaimers
│   │       └── orchestrator.py  # LegalAgent: multi-step orchestrator & execution trace
│   ├── data/                    # Legal Data Store (Git ignored)
│   │   ├── raw/                 # Destination for downloaded official Bare Act PDFs
│   │   ├── processed/           # Generated structured JSONL legal chunks
│   │   ├── manifests/           # source_manifest.json and source_manifest.csv
│   │   └── chroma_db/           # ChromaDB persistent on-disk vector database
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_health.py       # Health endpoint unit tests
│   │   ├── test_chat.py         # Chat endpoint validation, CORS, & agentic integration tests
│   │   ├── test_extract.py      # [STEP 2] PyMuPDF extraction & corrupt PDF tests
│   │   ├── test_clean.py        # [STEP 2] Legal whitespace & punctuation tests
│   │   ├── test_chunk.py        # [STEP 2] Section preservation & chunking tests
│   │   ├── test_metadata.py     # [STEP 2] Manifest provenance & CSV/JSON tests
│   │   ├── test_pipeline.py     # [STEP 2] End-to-end ingestion & idempotency tests
│   │   └── test_agent.py        # [STEP 4] 27 unit & integration tests for Planner, Synthesizer, & Agent
│   ├── requirements.txt         # FastAPI, Uvicorn, Pydantic, Pytest, HTTPX, PyMuPDF, ChromaDB, Sentence-Transformers
│   ├── .env.example             # Backend environment template
│   └── .env                     # Active backend environment configuration
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx           # Branding, live backend status ping, language selector
│   │   │   ├── ExampleQuestions.tsx # 4 legal pillars (Maintenance, Divorce, Custody, DV)
│   │   │   ├── ChatArea.tsx         # Conversation stream with AgentTracePanel & SourcesPanel
│   │   │   ├── ChatInput.tsx        # Large auto-growing textarea, validation, send button
│   │   │   ├── VoiceModal.tsx       # Placeholder modal detailing Step 5 ASR voice pipeline
│   │   │   └── DisclaimerBanner.tsx # Statutory Indian legal disclaimer
│   │   ├── constants/
│   │   │   └── translations.ts  # Localized strings & legal questions in EN, HI, KN
│   │   ├── services/
│   │   │   └── api.ts           # Typed API client for /api/health and /api/chat
│   │   ├── types.ts             # TypeScript definitions (Chat, RetrievedPassage, AgentStep)
│   │   ├── App.tsx              # Main interactive application state & container
│   │   ├── main.tsx             # React DOM entrypoint
│   │   └── index.css            # Tailwind CSS v4 styling & typography tokens
│   ├── package.json             # React 19, Vite, Tailwind CSS v4, Lucide Icons
│   ├── vite.config.ts           # Vite + React + Tailwind v4 plugin configuration
│   ├── .env.example             # Frontend environment template
│   └── .env                     # Active frontend environment configuration (VITE_API_BASE_URL)
│
├── .gitignore                   # Excludes venv, node_modules, raw PDFs, JSONL, ChromaDB, build artifacts
├── .env.example                 # Root configuration references
└── README.md                    # Project documentation & macOS run instructions
```

---

## Prerequisites for macOS

- **macOS**: Sonoma / Sequoia or newer (Apple Silicon M-series or Intel)
- **Python**: Version 3.10+ (Tested on Python 3.13)
- **Node.js**: Version 18+ (Tested on Node v26 / npm 11)

Verify your environment:
```bash
python3 --version
node --version
npm --version
```

---

## Quick Start (Step-by-Step macOS Commands)

### 1. Set Up and Run the Backend (FastAPI)

```bash
# Navigate to the backend directory
cd nyayasetu-ai/backend

# Create a virtual environment (if not already created)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install all backend dependencies
pip install -r requirements.txt

# Run all automated tests (55 tests across Steps 1-4)
PYTHONPATH=. pytest tests -v

# Start the FastAPI server with live reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend is accessible at:
- **API Base URL**: `http://127.0.0.1:8000/api`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **Health Check Endpoint**: `http://127.0.0.1:8000/api/health`
- **Corpus Status Endpoint**: `http://127.0.0.1:8000/api/corpus/status`
- **Chat Endpoint**: `POST http://127.0.0.1:8000/api/chat`
- **Passage Retrieval Endpoint**: `POST http://127.0.0.1:8000/api/retrieve`

---

### 2. Set Up and Run the Frontend (React + Vite)

In a separate terminal window:

```bash
# Navigate to the frontend directory
cd nyayasetu-ai/frontend

# Install dependencies (if not already installed)
npm install

# Build check (0 TypeScript errors)
npm run build

# Start the Vite development server
npm run dev
```

The frontend development server launches at:
- **Local Application URL**: `http://127.0.0.1:5173` (or `http://localhost:5173`)

---

## Step 4: LLM-Based Legal Answer Generation using RAG

Step 4 implements transparent, evidence-grounded legal answer generation powered by configurable LLMs (Local Ollama, OpenAI-compatible APIs, or zero-cost deterministic template synthesis).

### 1. Reusable & Configurable LLM Service (`app/services/llm_service.py`)
- **Zero-Cost Local Default**: By default, `LLM_PROVIDER=template`, running deterministic grounded synthesis with **0 external API keys and 0 paid subscriptions**.
- **Local Ollama Integration**: Set `LLM_PROVIDER=ollama` and `LLM_MODEL_NAME=llama3.2` (or `mistral`, `gemma2`) to run 100% locally on your machine via `http://localhost:11434`.
- **Hosted Provider Support**: Supports OpenAI, Groq, OpenRouter, DeepSeek, and Google Gemini via standard OpenAI-compatible endpoints (`LLM_PROVIDER=openai`, `LLM_API_KEY=...`, `LLM_API_BASE=...`).
- **Configurable Parameters**: Temperature (`LLM_TEMPERATURE=0.2`), token ceiling (`LLM_MAX_TOKENS=1024`), and request timeout (`LLM_REQUEST_TIMEOUT=30.0`).
- **Resilient Fallback**: If an external LLM fails due to network timeout or provider outage, the service automatically falls back to deterministic grounded synthesis without crashing.

### 2. Answer-Generation Pipeline & Citation Validation
1. **Query & Emergency Screening**: Incoming queries are screened by `SafetyAuditor` for immediate physical danger. Acute threats trigger emergency helpline routing (112, 181, 1091) instead of standard legal retrieval.
2. **Legal Retrieval**: ChromaDB is queried across generated sub-queries to retrieve relevant statutory provisions.
3. **Evidence Sufficiency Check**: If no indexed passages meet the minimum similarity threshold (`MIN_SIMILARITY_SCORE=0.25`), the system returns a controlled **Insufficient Evidence** response, refusing to speculate with unverified general knowledge.
4. **Structured Legal Answer Format**:
   - **Direct Explanation**: Objective, straightforward response to the user query.
   - **Relevant Legal Provisions**: Exact Act and Section numbers identified in the evidence.
   - **Plain-Language Interpretation**: Everyday explanation of statutory rights and procedures.
   - **Important Limitations & Conditions**: Jurisdictional prerequisites and procedural limits.
   - **Source Citations**: Numbered references (`[1]`, `[2]`) linked directly to official India Code bare acts.
5. **Citation Validation**: The synthesizer validates all cited section numbers against the retrieved evidence and generates warnings if hallucinated provisions are detected.

### 3. Immediate Danger & Safety Routing (`app/agent/safety.py`)
When a user describes imminent physical violence, threats to life, or crisis situations, the system bypasses ordinary retrieval and prioritizes official, verified Indian emergency helplines:
- **112** — National Emergency Response Support System (Police / Ambulance)
- **181** — Women Helpline (Ministry of WCD — 24x7 Domestic Abuse & Distress)
- **1091** — Women in Distress Police Helpline
- **7827170170** — National Commission for Women (NCW 24x7 Helpline)
- **1098** — Childline India (Minors in distress)
- **15100** — National Legal Services Authority ([NALSA](https://nalsa.gov.in/) Free Legal Aid)

### 4. Interactive Frontend Panels (`src/components/ChatArea.tsx`)
- **Evidence Status Badges**: Messages display `✓ Evidence Grounded`, `⚠️ Insufficient Evidence`, `📁 Knowledge Base Unindexed`, or `🚨 Immediate Safety Priority`.
- **Verified Citations Bar**: Top-level clickable badges for every verified statutory source referencing the official India Code repository.
- **Collapsible Sources Panel (`SourcesPanel`)**: Displays full statutory text, section numbers, page numbers, similarity scores, and outbound links.
- **Agent Reasoning Trace (`AgentTracePanel`)**: Shows concise step-by-step progress and timing in milliseconds without exposing private model chain-of-thought.

---

## Step 3: Retrieval-Augmented Generation (RAG) Retrieval Engine

Step 3 provides a high-precision, semantic retrieval engine that indexes processed legal chunks and retrieves authoritative statutory provisions matching user inquiries in **English**, **Hindi**, and **Kannada**.

### 1. Pretrained Multilingual Embedding Model
- **Model Identifier**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Source**: [Hugging Face Hub](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **License**: Apache 2.0 (open-source, permissible commercial/academic usage)
- **Dimensions**: 384 dimensions
- **Parameters / Size**: ~471 MB (118M parameters)
- **Language Support**: 50+ languages trained via multilingual knowledge distillation. Specifically verified for English (`en`), Hindi (`hi`), and Kannada (`kn`).
- **Vector Normalization**: Vectors are L2 unit-sphere normalized (`normalize_embeddings=True`), allowing cosine distance and dot products to directly measure semantic similarity.
- **Offline / Failure Resilience**: Lazy-loaded with an automatic deterministic pseudo-embedding fallback so unit tests and offline environments run reliably without crashing.
- **Model Policy**: The embedding model is strictly pretrained; no domain fine-tuning was performed.

### 2. Dedicated ChromaDB Vector Database
- **Backend Technology**: [ChromaDB](https://www.trychroma.com/) with persistent on-disk storage.
- **Storage Location**: `nyayasetu-ai/backend/data/chroma_db/`
- **Collection Name**: `nyayasetu_legal_corpus` (configurable via `CHROMA_COLLECTION_NAME` in `.env`)
- **Distance Metric**: Cosine similarity space (`hnsw:space: "cosine"`)
- **Idempotency & Upsert**: Indexed records use deterministic `chunk_id` values (e.g. `hma_1955_sec_13_1`) to prevent duplicate insertions on repeated indexing runs.
- **Index Rebuild Support**: Supports explicit rebuilds via `reset_collection()` / `--reset` flag without automatically deleting data unexpectedly.
- **Metadata Preservation**: Stores complete statutory provenance (`document_id`, `document_title`, `act_name`, `section_number`, `section_title`, `subsection_id`, `page_start`, `page_end`, `source_url`, `publication_date`, `effective_date`, `document_version`, `content_hash`).

### 3. Reusable Retrieval Service (`app/rag/retrieval.py`)
Exposes the core reusable retrieval function:
```python
from app.rag.retrieval import retrieve_legal_passages

response = retrieve_legal_passages(
    query="What are the grounds for divorce under Hindu law?",
    top_k=5,
    language="en",          # Supported: 'en', 'hi', 'kn'
    document_id=None,       # Optional filter, e.g. 'hma_1955'
    min_score=0.25,         # Configurable minimum similarity threshold
)
```

Each returned `RetrievedPassage` object contains:
- `chunk_id`: Deterministic unique chunk identifier.
- `legal_text`: Full statutory text slice.
- `document_title`: Formal title of the enactment.
- `act_name`: Common statutory name (e.g. *Hindu Marriage Act, 1955*).
- `section_number`: Statutory section number (e.g. `13`, `125`, `13B`).
- `section_title`: Section heading (e.g. *Divorce*, *Maintenance*).
- `page_start` & `page_number`: 1-indexed page in the official PDF bare act.
- `source_url`: Official [India Code](https://www.indiacode.nic.in/) bare act repository URL.
- `score`: Cosine similarity score (0.0 to 1.0; *not a probability of legal correctness*).
- `metadata`: Complete dictionary of all indexed statutory fields.

### 4. Indexing Command
To index all processed legal chunks from `data/processed/` into ChromaDB:
```bash
cd nyayasetu-ai/backend
source venv/bin/activate

# Standard idempotent indexing (upserts new or modified chunks)
python -m app.rag.indexer

# Clean rebuild (resets ChromaDB collection before re-indexing)
python -m app.rag.indexer --reset

# Custom batch size
python -m app.rag.indexer --batch-size 50
```
*Note: If no raw PDFs have been downloaded into `data/raw/` yet, the indexer gracefully reports that the corpus is empty and exits.*

### 5. Direct Retrieval Command & API Example
#### Via cURL (Direct HTTP API):
```bash
curl -X POST http://127.0.0.1:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the legal provisions related to maintenance?",
    "top_k": 5,
    "language": "en"
  }'
```

#### Example JSON Response:
```json
{
  "status": "success",
  "query": "What are the legal provisions related to maintenance?",
  "language": "en",
  "total_results": 1,
  "insufficient_evidence": false,
  "evidence_note": null,
  "passages": [
    {
      "chunk_id": "crpc_sec_125_sec_125_1",
      "document_id": "crpc_sec_125",
      "document_title": "Code of Criminal Procedure, 1973",
      "act_name": "Code of Criminal Procedure, 1973",
      "section_number": "125",
      "section_title": "Order for maintenance of wives, children and parents",
      "page_start": 15,
      "page_end": 16,
      "page_number": 15,
      "source_url": "https://www.indiacode.nic.in/handle/123456789/1611",
      "publication_date": "1974-01-25",
      "legal_text": "Section 125. Order for maintenance of wives, children and parents...",
      "score": 0.7421,
      "is_split_fallback": false,
      "metadata": { ... }
    }
  ],
  "disclaimer": "Retrieved passages reflect statutory provisions from indexed official sources. Similarity scores measure semantic closeness and must not be interpreted as calibrated probabilities of legal validity or outcomes."
}
```

### 6. Frontend Development Retrieval Mode
The React interface features a prominent mode toggle in the action bar:
- **`⚡ RAG Retrieval Mode (Step 3)`**: Submits queries directly to `POST /api/retrieve`. Displays the retrieved statutory passages, document titles, section numbers, page numbers, similarity scores, and India Code source URLs in the `SourcesPanel` without synthesizing an LLM answer. If no passages satisfy the relevance threshold, an informative "No Relevant Evidence Found" alert banner is displayed.
- **`🤖 Agentic Mode (Step 4)`**: Submits queries through the full agentic multi-hop planning and synthesis pipeline.

### 7. Evaluation Framework (`app/rag/evaluator.py`)
Computes standard information retrieval metrics against verified benchmark datasets:
- **Recall@5**: Fraction of queries where an expected relevant chunk appears in the top-5 retrieved passages.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant chunk ($1/\text{rank}$).
- **Retrieval Success Rate**: Percentage of queries with at least one matching hit in top-$k$.

#### Run Evaluation on Synthetic Documents (Testing Mechanics Only):
```bash
python -m app.rag.evaluator --run-synthetic --top-k 5
```
*Output:*
```
Total Benchmark Queries : 3
Verified Labeled Queries: 3
Recall@5              : 100.00%
Mean Reciprocal Rank (MRR): 1.0000
Retrieval Success Rate  : 100.00%
```

#### Run Evaluation on a Verified Ground-Truth Benchmark File:
```bash
python -m app.rag.evaluator --benchmark-file data/benchmarks/verified_eval.json --top-k 5
```

---

## Step 2: Legal Document Ingestion Pipeline

Step 2 extracts, cleans, chunks, and catalogs authoritative Indian family-law statutes into structured JSONL records for indexing.

### Initial Legal Corpus & India Code Sources

Bare Acts are sourced exclusively from [India Code](https://www.indiacode.nic.in/):

| Act / Provision Name | Canonical Act ID | Expected Local Filename | Authoritative Official URL |
| :--- | :---: | :--- | :--- |
| **Hindu Marriage Act, 1955** | `hma_1955` | `hindu_marriage_act_1955.pdf` | [India Code HMA 1955](https://www.indiacode.nic.in/handle/123456789/1567) |
| **Special Marriage Act, 1954** | `sma_1954` | `special_marriage_act_1954.pdf` | [India Code SMA 1954](https://www.indiacode.nic.in/handle/123456789/1547) |
| **Hindu Adoptions and Maintenance Act, 1956** | `hama_1956` | `hindu_adoptions_and_maintenance_act_1956.pdf` | [India Code HAMA 1956](https://www.indiacode.nic.in/handle/123456789/1640) |
| **Protection of Women from Domestic Violence Act, 2005** | `pwdva_2005` | `protection_of_women_from_domestic_violence_act_2005.pdf` | [India Code PWDVA 2005](https://www.indiacode.nic.in/handle/123456789/2085) |
| **Guardians and Wards Act, 1890** | `gwa_1890` | `guardians_and_wards_act_1890.pdf` | [India Code GWA 1890](https://www.indiacode.nic.in/handle/123456789/2296) |
| **CrPC 1973 (Sec 125-128 Maintenance) / BNSS 2023** | `crpc_sec_125` | `crpc_section_125_maintenance.pdf` | [India Code CrPC 1973](https://www.indiacode.nic.in/handle/123456789/1611) |
| **Family Courts Act, 1984** | `fca_1984` | `family_courts_act_1984.pdf` | [India Code FCA 1984](https://www.indiacode.nic.in/handle/123456789/1944) |

### How to Ingest & Index Official Legal Documents

1. **Download Official PDFs**:
   Visit the links above on [India Code](https://www.indiacode.nic.in/) and download the official PDF bare acts.
2. **Place in Raw Directory**:
   ```bash
   nyayasetu-ai/backend/data/raw/
   ```
3. **Run Ingestion Pipeline**:
   ```bash
   cd nyayasetu-ai/backend
   source venv/bin/activate
   python -m app.ingestion.pipeline
   ```
4. **Run RAG Indexer**:
   ```bash
   python -m app.rag.indexer
   ```
5. **Verify Indexing Status**:
   ```bash
   curl http://127.0.0.1:8000/api/corpus/status
   ```

---

## Running Automated Tests

To run the complete test suite across Steps 1 through 4:

```bash
cd nyayasetu-ai/backend
source venv/bin/activate
PYTHONPATH=. pytest tests -v
```

### Test Coverage (79 Tests Total — All Passing)

- **Step 1 (API Foundation)**:
  - `tests/test_health.py` (1 test): Health endpoint (`GET /api/health`).
  - `tests/test_chat.py` (7 tests): Chat endpoint validation, CORS, language parameters, and error handling (`POST /api/chat`).
- **Step 2 (Ingestion Pipeline)**:
  - `tests/test_extract.py` (5 tests): PyMuPDF extraction, 0-byte recovery, corrupt PDF handling, scanned PDF detection.
  - `tests/test_clean.py` (5 tests): Whitespace normalization, section preservation, de-hyphenation, header/footer removal.
  - `tests/test_chunk.py` (4 tests): Legal section boundary preservation, long section fallback, metadata persistence.
  - `tests/test_metadata.py` (4 tests): Canonical manifest registry, JSON/CSV sync, case-insensitive filename resolution.
  - `tests/test_pipeline.py` (2 tests): Idempotent batch processing, change detection, end-to-end ingestion on sample PDF.
- **Step 3 (RAG Retrieval Engine)**:
  - `tests/test_rag.py` (24 tests):
    1. Embedding generation: 384 dimensions, L2 normalization ($\approx 1.0$), English/Hindi/Kannada encoding, batch encoding, and mock fallback.
    2. Vector indexing: indexing synthetic chunks into ChromaDB, empty list handling.
    3. Repeated indexing: idempotency (no duplicate count), chunk updating via upsert, explicit collection reset.
    4. Metadata preservation: all statutory metadata fields stored and verified upon retrieval.
    5. Top-k retrieval: top-k cutoff enforcement, reusable `retrieve_legal_passages` function, metadata filtering by `document_id`.
    6. Empty queries: validation and rejection of empty/whitespace queries.
    7. Empty collections: graceful status and clear message when vector store contains 0 records.
    8. API request validation: Pydantic field constraints on `POST /api/retrieve` (422 responses on empty query, missing fields, out-of-range `top_k`, invalid language).
    9. Retrieval response structure: schema conformance, `RetrievedPassage` typing, and mandatory statutory disclaimer.
- **Step 4 (Agentic AI Pipeline)**:
  - `tests/test_agent.py` (27 tests): Intent classification across 7 categories in EN/HI/KN, sub-query generation, empty corpus fallback, grounded citation synthesis, low-score filtering, orchestrator step timing and traces, `/api/retrieve` endpoint, and `/api/corpus/status` endpoint.

---

## Roadmap Ahead

- **Step 1** (Completed): Frontend, FastAPI backend, CORS, schemas, and multilingual foundation.
- **Step 2** (Completed): Legal document ingestion, PyMuPDF extraction, conservative cleaning, section-aware chunking, and source manifests.
- **Step 3** (Completed): Vector embeddings, persistent ChromaDB store, and multilingual legal retrieval.
- **Step 4** (Completed): Agentic multi-hop planning, evidence-grounded synthesis, explainable agent trace, and citation UI.
- **Step 5**: Multilingual speech recognition (Web Speech API / IndicConformer / Whisper) and Text-to-Speech integration.
- **Step 6**: Advanced multi-turn conversational memory, interactive legal clarification dialogues, and court form assistance.
