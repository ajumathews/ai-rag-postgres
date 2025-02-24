import psycopg2
from psycopg2.extras import RealDictCursor

connection = psycopg2.connect(
    dbname="vector_db",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

cursor = connection.cursor(cursor_factory=RealDictCursor)


cursor.execute("SELECT * FROM products")
dataset = cursor.fetchall()

for data in dataset:
    print(data) 

cursor.close()
connection.close()
