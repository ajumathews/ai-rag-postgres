CREATE EXTENSION IF NOT EXISTS vector

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    product_description VARCHAR,
    price NUMERIC(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE product_embeddings (
    product_embedding_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding VECTOR(768),
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

