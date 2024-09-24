import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = OllamaEmbeddings(
    base_url='https://ollama.dealwallet.com',  # Base URL for Ollama embeddings
    model="nomic-embed-text"                   # Model to use for embeddings
)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Print the contents of the documents for debugging
print("\n--- Document Contents ---")
all_docs = db._collection.get()  # Internal method to fetch all documents

# Check if documents exist
if not all_docs['documents']:
    print("No documents found in the database.")
else:
    for i, doc in enumerate(all_docs['documents']):
        print(f"Document {i+1}:\n{doc}\n")

# Define the user's question
query = "How did Juliet die?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.0},  # Adjusted threshold
)

# Retrieve relevant documents
relevant_docs_with_scores = retriever.invoke(query)

# Display the relevant results with metadata and scores
print("\n--- Relevant Documents with Scores ---")
if relevant_docs_with_scores:
    for i, (doc, score) in enumerate(relevant_docs_with_scores, 1):
        print(f"Document {i}:\n{doc.page_content}\nScore: {score}\n")
        print(f"Source: {doc.metadata['source']}\n")
else:
    print("No relevant documents retrieved. Consider checking your documents or query.")
