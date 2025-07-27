import json
import os
import re
from datetime import datetime
from tqdm import tqdm
from utils import PDFProcessor

def load_input(input_path="input/input.json"):
    with open(input_path) as f:
        return json.load(f)

def process_documents(input_data, pdf_processor, input_dir="input"):
    documents = []
    for doc in tqdm(input_data["documents"], desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            continue
        sections = pdf_processor.extract_sections(pdf_path)
        documents.append({
            "filename": doc["filename"],
            "title": doc["title"],
            "sections": sections
        })
    return documents

def rank_content(documents, pdf_processor, persona, job):
    query_embedding = pdf_processor.create_context_embedding(persona, job)
    all_sections = []
    all_paragraphs = []
    for doc in documents:
        for section in doc["sections"]:
            if not section["title"] or len(section["title"].strip()) < 3:
                continue
            paragraphs = pdf_processor.extract_paragraphs(section["text"])
            for para in paragraphs:
                semantic_score = pdf_processor.calculate_semantic_similarity(para, query_embedding)
                content_relevance = pdf_processor.calculate_content_relevance(para, job)
                combined_score = semantic_score * content_relevance
                all_paragraphs.append({
                    "document": doc["filename"],
                    "text": para,
                    "page": section["page"],
                    "section_title": section["title"],
                    "relevance": combined_score
                })
            semantic_score = pdf_processor.calculate_semantic_similarity(
                section["title"] + " " + section["text"], 
                query_embedding
            )
            content_relevance = pdf_processor.calculate_content_relevance(
                section["title"] + " " + section["text"],
                job
            )
            combined_score = semantic_score * content_relevance
            all_sections.append({
                "document": doc["filename"],
                "title": section["title"],
                "page": section["page"],
                "relevance": combined_score
            })
    return all_sections, all_paragraphs

def generate_output(input_data, ranked_sections, ranked_paragraphs):
    selected_sections = []
    seen_docs = set()
    for section in sorted(ranked_sections, key=lambda x: x["relevance"], reverse=True):
        if section["document"] not in seen_docs:
            clean_title = section["title"]
            clean_title = re.sub(r'^\W+', '', clean_title)
            clean_title = re.sub(r'\s+', ' ', clean_title)
            clean_title = clean_title[:100]
            selected_sections.append({
                "document": section["document"],
                "title": clean_title,
                "page": section["page"],
                "relevance": section["relevance"]
            })
            seen_docs.add(section["document"])
        if len(selected_sections) >= 5:
            break
    selected_paragraphs = []
    seen_docs = set()
    for para in sorted(ranked_paragraphs, key=lambda x: x["relevance"], reverse=True):
        if para["document"] not in seen_docs:
            clean_text = para["text"]
            clean_text = re.sub(r'^\W+', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            if any(verb in clean_text.lower() for verb in 
                  {"visit", "explore", "try", "go", "see", "stay", "eat", "book", "reserve", "discover"}):
                selected_paragraphs.append({
                    "document": para["document"],
                    "text": clean_text,
                    "page": para["page"],
                    "relevance": para["relevance"]
                })
                seen_docs.add(para["document"])
        if len(selected_paragraphs) >= 5:
            break
    return {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [{
            "document": s["document"],
            "section_title": s["title"],
            "importance_rank": i+1,
            "page_number": s["page"]
        } for i, s in enumerate(selected_sections)],
        "subsection_analysis": [{
            "document": p["document"],
            "refined_text": p["text"],
            "page_number": p["page"]
        } for p in selected_paragraphs]
    }

def save_output(output, path="output/output.json"):
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

def main():
    pdf_processor = PDFProcessor()
    input_data = load_input()
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    documents = process_documents(input_data, pdf_processor)
    ranked_sections, ranked_paragraphs = rank_content(documents, pdf_processor, persona, job)
    output = generate_output(input_data, ranked_sections, ranked_paragraphs)
    save_output(output)

if __name__ == "__main__":
    main()