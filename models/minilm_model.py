from sentence_transformers import SentenceTransformer

def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
