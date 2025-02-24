import ollama
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

input_text = "Give me laptops with Nvidia graphics card"

response = ollama.embeddings(model="nomic-embed-text", prompt=input_text)

embeddings :list[float] = response.get('embedding', None) 

query = f"""
WITH vector_search AS(
SELECT pe.product_id, 
       RANK() OVER (ORDER BY pe.embedding <=> %s) AS rank
FROM product_embeddings pe
ORDER BY pe.embedding <=> %s
LIMIT 20
)
--select * from vector_search

,fulltext_search AS(
SELECT product_id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', product_description), query) DESC)
  FROM products, plainto_tsquery('english', %s) query
  WHERE to_tsvector('english', product_description) @@ query
  ORDER BY ts_rank_cd(to_tsvector('english', product_description), query) DESC
  LIMIT 20
)

--select * from fulltext_search

,hybrid_search AS(
SELECT
  COALESCE(vector_search.product_id, fulltext_search.product_id) AS product_id,
  COALESCE(1.0 / (60 + vector_search.rank), 0.0) +
  COALESCE(1.0 / (60 + fulltext_search.rank), 0.0) AS score
FROM vector_search
FULL OUTER JOIN fulltext_search ON vector_search.product_id = fulltext_search.product_id
ORDER BY score DESC
LIMIT 20
)

select 
    DISTINCT hs.product_id, 
    p.product_name,
    p.product_description,
    p.price,
    hs.score
from hybrid_search hs
join products p on hs.product_id = p.product_id
order by hs.score desc
"""


connection = psycopg2.connect(
    dbname="vector_db",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

cursor = connection.cursor(cursor_factory=RealDictCursor)

vector_embeddings = f'[{", ".join(map(str, embeddings))}]'

cursor.execute(query, (vector_embeddings, vector_embeddings, input_text))
dataset = cursor.fetchall()

formatted_sources = ""
for data in dataset[:5]:
    productId = data['product_id']
    product_name = data['product_name']
    price = data['price']
    description = data['product_description']
    formatted_sources += f"[{productId}]: Name: {product_name} Description: {description} Price:{price}\n\n"

cursor.close()
connection.close()

system_prompt = """
Assistant helps customers with questions about products.
Respond as if you are a salesperson helping a customer in a store.
Do NOT respond with tables.Answer ONLY with the product details listed in the products.
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Each product has an ID in brackets followed by colon and the product details.
Always include the product ID for each product you use in the response.
Use square brackets to reference the source, for example [52].\nDon't combine citations, list each product separately, for example [27][51].
"""
user_message = input_text
sources_section = "\nSources:\n"
sources_section += formatted_sources
user_prompt = user_message + sources_section


print("System Prompt:", system_prompt)
print("User Prompt:", user_prompt)
# Send the prompt and message to Ollama
response = ollama.chat(model="llama3.2",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        options={"temperature":0})
print("Response:", response['message'].content)