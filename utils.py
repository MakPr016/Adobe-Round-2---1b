import pdfplumber
import re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PDFProcessor:
    def __init__(self, domain_config=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.config = {
            "action_verbs": {"visit", "explore", "try", "go", "see", "stay", "eat", 
                             "book", "reserve", "create", "manage", "prepare", "develop"},
            "boost_terms": set(),
            "penalty_terms": set(),
            "heading_threshold": 1.5,
            "min_heading_length": 3,
            "max_heading_length": 100,
            "paragraph_min_length": 30,
            "paragraph_max_length": 500
        }
        if domain_config:
            for key in self.config:
                if key in domain_config:
                    if isinstance(self.config[key], set):
                        self.config[key] |= set(domain_config[key])
                    else:
                        self.config[key] = domain_config[key]
    
    def extract_sections(self, pdf_path):
        sections = []
        with pdfplumber.open(pdf_path) as pdf:
            current_section = None
            current_content = ""
            for page_num, page in enumerate(pdf.pages, 1):
                words = page.extract_words(extra_attrs=["size", "fontname"])
                if not words:
                    continue
                font_sizes = [w['size'] for w in words]
                median_font_size = np.median(font_sizes) if font_sizes else 10
                page_height = page.height
                lines = defaultdict(list)
                for word in words:
                    line_key = round(word['top'], 1)
                    lines[line_key].append(word)
                for y_pos, words_in_line in sorted(lines.items(), key=lambda x: x[0]):
                    if not words_in_line:
                        continue
                    line_text = " ".join(w['text'] for w in words_in_line)
                    avg_font_size = np.mean([w['size'] for w in words_in_line])
                    is_bold = any('bold' in w.get('fontname', '').lower() for w in words_in_line)
                    is_heading = (
                        (avg_font_size > median_font_size * self.config["heading_threshold"]) or
                        is_bold or
                        (y_pos < page_height * 0.15)
                    )
                    is_valid_heading = (
                        self.config["min_heading_length"] < len(line_text) < self.config["max_heading_length"] and
                        line_text.strip()[0].isupper()
                    )
                    if is_heading and is_valid_heading:
                        clean_title = line_text.strip()
                        clean_title = re.sub(r'^\W+', '', clean_title)
                        clean_title = re.sub(r'\s+', ' ', clean_title)
                        if current_section:
                            sections.append(current_section)
                        current_section = {
                            "title": clean_title,
                            "text": "",
                            "page": page_num
                        }
                        current_content = ""
                    elif current_section:
                        current_content += line_text + " "
                if current_section:
                    current_section["text"] += current_content
                    current_content = ""
            if current_section:
                sections.append(current_section)
        return sections

    def extract_paragraphs(self, text):
        if not text:
            return []
        raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        cleaned_paragraphs = []
        for para in raw_paragraphs:
            clean_para = re.sub(r'[\u2022\u25CF\*\-\+]', '', para)
            clean_para = re.sub(r'\s+', ' ', clean_para).strip()
            clean_para = re.sub(r'^\W+', '', clean_para)
            if self.config["paragraph_min_length"] < len(clean_para) < self.config["paragraph_max_length"]:
                cleaned_paragraphs.append(clean_para)
        return cleaned_paragraphs

    def calculate_semantic_similarity(self, text, query_embedding):
        if not text:
            return 0.0
        text_embedding = self.model.encode([text])
        return cosine_similarity([query_embedding], text_embedding)[0][0]

    def create_context_embedding(self, persona, job):
        return self.model.encode([f"Persona: {persona}\nTask: {job}"])[0]
    
    def calculate_content_relevance(self, text, job):
        if not text:
            return 1.0
        text_lower = text.lower()
        job_lower = job.lower()
        score = 1.0
        if any(verb in text_lower for verb in self.config["action_verbs"]):
            score *= 1.5
        for term in self.config["boost_terms"]:
            if term.lower() in text_lower:
                score *= 1.3
        for term in self.config["penalty_terms"]:
            if term.lower() in text_lower:
                score *= 0.7
        numbers = re.findall(r'\b\d+\b', job)
        for num in numbers:
            if num in text:
                score *= 1.5
        return min(3.0, score)