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
    for doc in tqdm(input_data["documents"], desc="Processing PDFs", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
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
            full_text = section["title"] + " " + section["text"]
            semantic_score = pdf_processor.calculate_semantic_similarity(full_text, query_embedding)
            content_relevance = pdf_processor.calculate_content_relevance(full_text, job)
            combined_score = semantic_score * content_relevance
            all_sections.append({
                "document": doc["filename"],
                "title": section["title"],
                "page": section["page"],
                "relevance": combined_score
            })
    return all_sections, all_paragraphs

def generate_output(input_data, ranked_sections, ranked_paragraphs):
    selected_sections = sorted(ranked_sections, key=lambda x: x["relevance"], reverse=True)[:5]
    selected_paragraphs = sorted(ranked_paragraphs, key=lambda x: x["relevance"], reverse=True)[:5]
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

def create_domain_config(job_description):
    job_lower = job_description.lower()
    keywords = set(re.findall(r'\b\w+\b', job_lower))
    stop_words = {"and", "the", "for", "to", "of", "in", "a", "on", "with", "as", "be", "is"}
    keywords = keywords - stop_words
    action_verbs = {
        "create", "manage", "prepare", "plan", "organize", 
        "develop", "build", "make", "design", "arrange"
    }
    detected_verbs = {verb for verb in action_verbs if verb in job_lower}
    return {
        "action_verbs": detected_verbs,
        "boost_terms": keywords,
        "penalty_terms": set(),
        "heading_threshold": 1.5
    }

def main():
    input_data = load_input()
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    domain_config = create_domain_config(job)
    pdf_processor = PDFProcessor(domain_config)
    documents = process_documents(input_data, pdf_processor)
    ranked_sections, ranked_paragraphs = rank_content(documents, pdf_processor, persona, job)
    output = generate_output(input_data, ranked_sections, ranked_paragraphs)
    save_output(output)

if __name__ == "__main__":
    main()