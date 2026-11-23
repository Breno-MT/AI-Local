from llama_cpp import Llama
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=0,
    verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how GGUF models work in two sentences."}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
