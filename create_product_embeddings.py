import psycopg2
import ollama
from psycopg2.extras import RealDictCursor
import numpy as np  # To handle the embedding as a vector

# Setup PostgreSQL connection
conn = psycopg2.connect(
    dbname="vector_db",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

# Create a cursor to interact with the database
cur = conn.cursor(cursor_factory=RealDictCursor)

# Fetch all products from the products table
cur.execute("SELECT product_id, product_name, product_description FROM products")
products = cur.fetchall()

# Iterate over each product and generate the embedding using Ollama
for product in products:
    # Concatenate the product name and description to form the input text
    ## Add a question to the embedding prompt 
    product_for_embedding = f"Product Name: {product['product_name']} Description: {product['product_description']}"
    response = ollama.embeddings(model="nomic-embed-text", prompt=product_for_embedding)
    embedding = response['embedding']  
    embedding_vector = np.array(embedding, dtype=np.float32)

    if embedding_vector.shape[0] != 768:
        raise ValueError(f"Embedding has incorrect shape: {embedding_vector.shape}. Expected 768.")

    # Insert or update the product embedding in the products_embedding table
    cur.execute("""
        INSERT INTO product_embeddings (product_id, embedding)
        VALUES (%s, %s)""", (product['product_id'], embedding_vector.tolist()))

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()
