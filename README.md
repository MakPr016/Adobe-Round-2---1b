# Adobe Hackathon - PDF Intelligence Engine

This project is a solution for the Adobe "Connecting the Dots" Hackathon, Round 1B: Persona-Driven Document Intelligence.

## 🔍 Overview

This solution analyzes a collection of PDFs based on a provided persona and their job-to-be-done. It extracts and ranks the most relevant sections and paragraphs using semantic similarity and content relevance.

## 🧠 Approach

1. **PDF Parsing:** We use `pdfplumber` to extract sections based on headings.
2. **Embedding & Relevance:** The `all-MiniLM-L6-v2` model from `sentence-transformers` (≈80MB) is used to create semantic embeddings.
3. **Ranking:** Relevance is scored by a combined metric using semantic similarity and domain-specific keywords from the job description.
4. **Output:** A structured JSON output includes top 5 sections and paragraphs based on importance.

## 🛠️ Models and Libraries Used

- `pdfplumber`: For parsing PDF content.
- `sentence-transformers`: For semantic embeddings (`all-MiniLM-L6-v2` model).
- `scikit-learn`: For cosine similarity computation.
- `tqdm`: For progress bars.
- `numpy`: For array computations.

> Total model size used: ~80 MB ✔️ (compliant with ≤1GB limit)
> Total execution time on 3–5 PDFs: Under 60 seconds ✔️

## 🐳 How to Build and Run (for documentation)

To build:
```bash
docker build --platform linux/amd64 -t mysolution:latest .
```
To run:
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolution:latest
```

Make sure:
- Your input PDFs are placed in /input
- Output JSONs will be created in /output

## ✅ Features
- Supports arbitrary PDF sets (3–10 docs)
- Ranks relevance using both semantic and rule-based scoring
- Modular and efficient: fits CPU-only 10s constraint
- Compliant with hackathon constraints (offline, model size <1GB, CPU-only)

## 📁 Directory Structure
```
├── main.py               # Orchestration script
├── utils.py              # PDFProcessor class for extraction and ranking
├── Dockerfile            # Docker container setup
├── input/                # Folder to place PDF files and input.json
├── output/               # Output JSON will be saved here
```

## 📌 Constraints Satisfied
- [x] CPU-only, no GPU
- [x] Model size < 1GB
- [x] No internet access
- [x] Execution time under 60s
- [x] Docker compatible with AMD64