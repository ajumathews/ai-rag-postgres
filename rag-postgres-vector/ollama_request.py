import ollama

# Set up the system prompt and user message
system_prompt = "You are a helpful assistant."
user_message = "Hello, how are you today?"

# Send the prompt and message to Ollama
response = ollama.chat(model="llama3.2",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])

# Print the response from Ollama
assistant_message = response['message'].content

# Print the assistant's reply
print("Response from Ollama:", assistant_message)