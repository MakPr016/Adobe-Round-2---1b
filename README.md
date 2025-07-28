
# Adobe Hackathon - Persona-Driven Document Intelligence

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

---

## 🚀 How to Run the Project

### Option 1: Using Docker (Recommended for Hackathon)

To build:
```bash
docker build --platform linux/amd64 -t mysolution:latest .
```

To run:
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolution:latest
```

Before running, copy your files to the `input/` folder:
```bash
cp /path/to/your/files/* ./input/
```

Make sure:
- Your input PDFs and `input.json` are in the `input/` folder.
- Output JSON will be created in the `output/` folder.

---

### Option 2: Run Locally Without Docker

#### 1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Place input files:
Make sure your PDFs and `input.json` are placed in the `input/` directory.

#### 3. Run the script:
```bash
python main.py
```

The output will be saved in the `output/` folder.

---

## 📁 Copying Files (if needed)

If you need to copy input/output files from one directory to another, you can use:

```bash
# Syntax: cp [source] [destination]
cp -r ./input /desired/path/
cp -r ./output /desired/path/
```

Or from outside into the repo:
```bash
cp /source/folder/*.pdf ./input/
```

---

## ✅ Features

- Supports arbitrary PDF sets (3–10 docs)
- Ranks relevance using both semantic and rule-based scoring
- Modular and efficient: fits CPU-only 10s constraint
- Compliant with hackathon constraints (offline, model size <1GB, CPU-only)

---

## 📁 Directory Structure
```
├── main.py               # Orchestration script
├── utils.py              # PDFProcessor class for extraction and ranking
├── Dockerfile            # Docker container setup
├── requirements.txt      # Python package requirements
├── input/                # Folder to place PDF files and input.json
├── output/               # Output JSON will be saved here
```

## 🧠 Libraries and Models Used

| Library / Model           | Purpose                                          | Size            |
|---------------------------|--------------------------------------------------|-----------------|
| `pdfplumber`              | Extract structured text from PDFs               | ~1.5 MB         |
| `sentence-transformers`   | Generate semantic embeddings                    | ~80 MB (model)  |
| `all-MiniLM-L6-v2`        | Pretrained sentence embedding model             | ~80 MB          |
| `scikit-learn`            | Cosine similarity, ML utilities                 | ~7–10 MB        |
| `tqdm`                    | Visual progress bars                            | ~70 KB          |
| `numpy`                   | Numerical computation support                   | ~17 MB          |
| `pytest`, `pytest-cov`    | Testing and code coverage (for dev use)         | ~1.5 MB         |


## 📌 Constraints Satisfied

- [x] CPU-only, no GPU
- [x] Model size < 1GB
- [x] No internet access
- [x] Execution time under 60s
- [x] Docker compatible with AMD64
- [x] Also runs without Docker (pure Python mode)

---
