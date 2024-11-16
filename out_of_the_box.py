from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import BeautifulSoupWebReader

reader = BeautifulSoupWebReader()
documents = reader.load_data(["https://docs.llamaindex.ai/en/stable/getting_started/customization/"])

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2:1b", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

while True:
    query_str = input("Ask a question: ")
    if query_str == "exit":
        break
    response = query_engine.query(query_str)
    print("\nAssistant:")
    for text in response.response_gen:
        print(text, end="", flush=True)
    print("\n")