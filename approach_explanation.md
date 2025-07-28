
# Approach Explanation – Adobe Hackathon Round 1B
## Challenge: Persona-Driven Document Intelligence

---

## 🔍 Objective

The goal of this solution is to build an intelligent PDF analysis system that extracts and ranks the **most relevant sections and sub-sections** from a collection of documents based on a **given persona** and their **job-to-be-done**.

The system must understand both the structural and semantic content of documents and deliver meaningful summaries, tailored for each user type.

---

## 🧠 High-Level Strategy

This project is structured into the following core stages:

### 1. 📥 Input Parsing

- Load the **input JSON** that includes:
  - `documents`: List of PDF filenames and metadata
  - `persona`: User’s role and focus areas
  - `job_to_be_done`: The concrete task the persona wants to accomplish

### 2. 📄 PDF Section Extraction

- We use `pdfplumber` to parse PDF files.
- Each page is scanned line-by-line to detect **headings** and **text blocks**.
- Headings are identified using heuristics:
  - Font size greater than the **median**
  - Font name contains **bold**
  - Position near the **top of the page**
- Validated against configurable constraints like:
  - Heading length (min/max)
  - Starts with uppercase
- Each heading opens a new **section**, and following lines are appended as **section text**.

This creates a structured document with a list of sections:
```json
{
  "title": "Introduction to GNN",
  "text": "...",
  "page": 2
}
```

---

### 3. ✂️ Paragraph Extraction

- Section text is split into **paragraphs** based on:
  - Double newlines (`\n\n`)
  - Minimum and maximum length thresholds
  - Removal of bullet symbols and formatting noise
- Each paragraph is cleaned and prepared for semantic analysis.

---

### 4. 🧬 Semantic Embedding & Scoring

#### a. **Embedding Creation**
- A sentence embedding model (`all-MiniLM-L6-v2`) is loaded using `sentence-transformers`.
- A combined embedding is created for:
  ```
  Persona: [persona]
  Task: [job_to_be_done]
  ```

#### b. **Relevance Scoring**
Each paragraph and section is scored based on:
1. **Semantic similarity**:
   - Cosine similarity between the paragraph and persona+job embedding
2. **Content relevance**:
   - Boosts score if:
     - Contains **action verbs** (`prepare`, `analyze`, `study`)
     - Contains **keywords** from the job description
     - Includes **numeric indicators** (like dates, values)
   - Penalizes for irrelevant or noisy terms (if configured)

Combined relevance score:
```
final_score = semantic_similarity * content_relevance
```

---

### 5. 📊 Ranking & Output Generation

- Top **5 sections** and **5 paragraphs** are selected based on final scores.
- Output includes:
  - Metadata (persona, job, timestamp)
  - Extracted sections with titles, document, page, and rank
  - Subsection analysis: paragraph text and location

---

## 🧱 Domain Config Customization

The system dynamically builds a **domain config** from the job description:
- Extracts keywords (ignoring stopwords)
- Detects intent from action verbs
- These are passed into the `PDFProcessor` to influence scoring heuristics

---

## ✅ Constraints Met

| Constraint                      | Status       |
|-------------------------------|--------------|
| Model size < 1 GB             | ✅ (~80 MB)  |
| CPU-only execution            | ✅           |
| No network access             | ✅           |
| Execution time < 60s          | ✅ Tested    |
| Works for 3–10 PDFs           | ✅ Modular   |

---

## 🔁 Modular Components

- `main.py`: Loads inputs, handles pipeline flow
- `utils.py`: Contains `PDFProcessor` class with all NLP and PDF logic
- Dockerfile: Installs dependencies, ensures offline, CPU-only operation

---

## 📌 Summary

This solution combines traditional PDF parsing with semantic embeddings and configurable domain relevance to prioritize the content that **matters most for each user**.

The design ensures scalability, adaptability across domains, and compliance with all Adobe Hackathon constraints.
