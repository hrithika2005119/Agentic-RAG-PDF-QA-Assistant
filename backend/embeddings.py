import ollama

def load_model():
    model_name = "nomic-embed-text"

    models = ollama.list()

    for model in models["models"]:
        if model.model.startswith(model_name):
            return model_name
        
    raise RuntimeError(f"{model_name} is not available in Ollama.")

def generate_embeddings(chunks):
    model_name = load_model()

    embeddings = []

    for chunk in chunks:
        response = ollama.embed(
            model = model_name, 
            input = chunk
            )
        embeddings.append(response["embeddings"][0])

    return embeddings

if __name__ == "__main__":
    sample_chunks = ["Python is a programming language.", 
                     "Ollama runs AI models locally.", 
                     "Embeddings convert text into numbers."]
    
    embeddings = generate_embeddings(sample_chunks)

    print(f"Number of embeddings:{len(embeddings)}")
    print(f"First embedding:{embeddings[0]}")
    print(f"Vector length:{len(embeddings[0])}")