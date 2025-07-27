# Approach Explanation - Adobe Hackathon Round 1B

## Problem Summary

The goal of Round 1B is to process a collection of PDF documents and extract the most relevant sections and paragraphs based on a given persona and job-to-be-done. The system must intelligently understand the content and rank outputs in a way that aligns with user intent.

---

## Methodology

### 1. **PDF Parsing and Section Extraction**
We used `pdfplumber` to extract textual content along with layout metadata such as font size, boldness, and vertical position. This metadata is then analyzed to identify headings (likely to be sections). The logic uses a combination of:
- Relative font size (1.5× median)
- Bold font presence
- Position on the page (top 15%)
- Heuristic filters (e.g., no numbers, short and capitalized phrases)

Each heading begins a new "section", and the text beneath it until the next heading is captured as that section's content.

---

### 2. **Context Embedding Creation**
A semantic context is created from the persona and job description using a `SentenceTransformer` model (`all-MiniLM-L6-v2`). This embedding is used to compare with each extracted section/paragraph to calculate semantic relevance.

Additional task-specific clues (e.g., "group of 10", "college friends", "4 days") are also parsed from the job to further refine the context.

---

### 3. **Relevance Scoring and Ranking**
Each section and its paragraphs are ranked using:
- **Semantic similarity** with the context embedding
- **Content relevance** using rule-based keyword scoring (e.g., travel-related verbs, duration/group info, domain-specific boosts)

The final score is a multiplication of these two, ensuring both semantic and task-relevance are considered.

---

### 4. **Selection and Output**
- The top 5 **sections** are selected from different documents based on ranked relevance.
- The top 5 **paragraphs** are selected using additional filters (e.g., must contain action verbs).
- Output is formatted in JSON with metadata, section ranking, and paragraph analysis.

---

## Compliance
- ✅ No hardcoded rules or filenames
- ✅ All logic works offline (no internet calls)
- ✅ Model size is under 100MB (`MiniLM`)
- ✅ Runs efficiently on CPU (tested with 3–5 documents within 60 seconds)

---

## Improvements & Next Steps
- Add multilingual support
- Improve heading hierarchy (e.g., detect H1/H2/H3)
- Use lightweight layout models to infer visual structure

