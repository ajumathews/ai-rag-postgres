import ollama

# Example input text
input_text = "macbook"

# Call the Ollama API to generate embeddings
response = ollama.embeddings(model="nomic-embed-text", prompt=input_text)

# Extract embeddings from the response (if available)
embeddings = response.get('embedding', None)

if embeddings:
    print(f"Embeddings: {embeddings}")
else:
    print("No embeddings found in the response.")
