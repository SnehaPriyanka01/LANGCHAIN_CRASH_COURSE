import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model using Ollama
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# Function to query a vector store with different search types and parameters
def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

# Define the user's question
query = "How did Juliet die?"

# Showcase different retrieval methods

# 1. Similarity Search
print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embedding_model, "similarity", {"k": 3})

# 2. Max Marginal Relevance (MMR)
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embedding_model,
    "mmr",
    {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# 3. Similarity Score Threshold
print("\n--- Using Similarity Score Threshold ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embedding_model,
    "similarity_score_threshold",
    {"k": 3, "score_threshold": 0.1},
)

print("Querying demonstrations with different search types completed.")
