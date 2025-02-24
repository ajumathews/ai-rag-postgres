import ollama
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import json

def extract_search_arguments(response_message, original_user_query):
    search_query = None
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            function = tool.function
            if function.name == "search_database":
                json_string = json.dumps(function.arguments)
                arg = json.loads(json_string)
                search_query = arg.get("search_query", original_user_query)
                if "price_filter" in arg and arg["price_filter"]:
                    price_filter = arg["price_filter"]
                    filters.append(
                        {
                            "column": "price",
                            "comparison_operator": price_filter["comparison_operator"],
                            "value": price_filter["value"],
                        }
                    )
    elif query_text := response_message.content:
        search_query = query_text.strip()
    return search_query, filters


def build_filter_clause(filters) -> tuple[str, str]:
        if filters is None:
            return "", ""
        filter_clauses = []
        for filter in filters:
            if isinstance(filter["value"], str):
                filter["value"] = f"'{filter['value']}'"
            filter_clauses.append(f"{filter['column']} {filter['comparison_operator']} {filter['value']}")
        filter_clause = " AND ".join(filter_clauses)
        if len(filter_clause) > 0:
            return f"WHERE {filter_clause}", f"AND {filter_clause}"
        return "", ""


input_text = "Do you have Apple macbooks less than 2000 dollars"

query_prompt = """
Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching database rows.
You have access to an Azure PostgreSQL database with an items table that has columns for title, description, brand, price, and type.
Generate a search query based on the conversation and the new question.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return the original user question.
DO NOT return anything besides the query.

Few-shot examples:

1. User: "What are some red shoes?"
   Assistant: 
   {
       "id": "call_abc123",
       "type": "function",
       "function": {
           "arguments": {
               "search_query": "red shoes"
           },
           "name": "search_database"
       }
   }

2. User: "Do you have any shoes under $50?"
   Assistant: 
   {
       "id": "call_abc124",
       "type": "function",
       "function": {
           "arguments": {
               "search_query": "shoes",
               "price_filter": {
                   "comparison_operator": "<",
                   "value": 50
               }
           },
           "name": "search_database"
       }
   }

3. User: "Can you show me Nike running shoes?"
   Assistant: 
   {
       "id": "call_abc125",
       "type": "function",
       "function": {
           "arguments": {
               "search_query": "running shoes",
               "brand_filter": {
                   "comparison_operator": "=",
                   "value": "Nike"
               }
           },
           "name": "search_database"
       }
   }

4. User: "Are there any Adidas shoes under $100?"
   Assistant: 
   {
       "id": "call_abc126",
       "type": "function",
       "function": {
           "arguments": {
               "search_query": "shoes",
               "price_filter": {
                   "comparison_operator": "<",
                   "value": 100
               },
               "brand_filter": {
                   "comparison_operator": "=",
                   "value": "Adidas"
               }
           },
           "name": "search_database"
       }
   }

5. User: "Can you find laptops that cost at least $1000?"
   Assistant: 
   {
       "id": "call_abc127",
       "type": "function",
       "function": {
           "arguments": {
               "search_query": "laptops",
               "price_filter": {
                   "comparison_operator": ">=",
                   "value": 1000
               }
           },
           "name": "search_database"
       }
   }
"""


query_products_tool = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search PostgreSQL database for relevant products based on user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to use for full text search, e.g. 'red shoes'",
                        },
                        "price_filter": {
                            "type": "object",
                            "description": "Filter search results based on price of the product",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator to compare the column value, either '>', '<', '>=', '<=', '='",  # noqa
                                },
                                "value": {
                                    "type": "number",
                                    "description": "Value to compare against, e.g. 30",
                                },
                            },
                        },
                        "brand_filter": {
                            "type": "object",
                            "description": "Filter search results based on brand of the product",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator to compare the column value, either '=' or '!='",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Value to compare against, e.g. AirStrider",
                                },
                            },
                        },
                    },
                    "required": ["search_query"],
                },
            },
        }


response = ollama.chat(
    'llama3.2',
    messages=[{"role": "system", "content": query_prompt}, {"role": "user", "content": input_text}],
    options={"temperature":0},
    tools=[query_products_tool]
)

# Extract tool call details from the response
search_query, filters = extract_search_arguments(response['message'], input_text)
print("Search Query:", search_query)


filter_clause_where, filter_clause_and = build_filter_clause(filters)
print("Filter Clause Where:", filter_clause_where)
print("Filter Clause And:", filter_clause_and)

response = ollama.embeddings(model="nomic-embed-text", prompt=search_query)

embeddings :list[float] = response.get('embedding', None) 

query = f"""
WITH filtered_products AS (
    SELECT *
    FROM products
    {filter_clause_where}
),

vector_search AS(
SELECT pe.product_id, 
       RANK() OVER (ORDER BY pe.embedding <=> %s) AS rank
FROM product_embeddings pe
JOIN filtered_products fp ON pe.product_id = fp.product_id
ORDER BY pe.embedding <=> %s
LIMIT 20
)
--select * from vector_search

,fulltext_search AS(
SELECT product_id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', product_description), query) DESC)
  FROM filtered_products, plainto_tsquery('english', %s) query
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
join filtered_products p on hs.product_id = p.product_id
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

cursor.execute(query, (vector_embeddings, vector_embeddings, search_query))
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
                        options={"temperature":0.0})
print("Response:", response['message'].content)




