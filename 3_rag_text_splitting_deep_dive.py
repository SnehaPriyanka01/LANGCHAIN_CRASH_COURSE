import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

# Load or create the Chroma vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # Size of each chunk
    chunk_overlap=100,  # Overlap between chunks to ensure context continuity
    separator="\n",  # Split by newline characters
)

# Step 1: Check if documents exist in Chroma vector store
all_docs = db._collection.get()
if len(all_docs['documents']) == 0:
    print("No documents found in the vector store. Please make sure documents are added to the store.")

    # Example: Add documents to the vector store if none exist
    texts = [
        "Romeo and Juliet is a tragic love story.",
        "Juliet dies after taking a potion to avoid marrying Paris."
    ]
    metadata = [{"source": "doc1"}, {"source": "doc2"}]

    # Add these documents to the vector store
    db = Chroma.from_texts(texts, embeddings, metadatas=metadata, persist_directory=persistent_directory)

    # Persist the data
    db.persist()

    print("Documents added to the vector store. Please run the script again.")
else:
    print(f"Number of documents: {len(all_docs['documents'])}")

# Step 2: Split the documents into chunks
split_documents = []
for i, doc in enumerate(all_docs['documents']):
    # Split the document into smaller chunks
    chunks = text_splitter.split_text(doc)
    for chunk in chunks:
        split_documents.append({"text": chunk, "metadata": all_docs['metadatas'][i]})

# Print the split document chunks for debugging
print("\n--- Split Document Chunks ---")
for i, split_doc in enumerate(split_documents):
    print(f"Chunk {i+1}:\n{split_doc['text']}\nMetadata: {split_doc['metadata']}\n")

# Step 3: Define the user's question
query = "How did Juliet die?"

# Step 4: Check the embedding model for query
query_embedding = embeddings.embed_query(query)  # Fix: Use embed_query instead of embed_text
print(f"Query Embedding: {query_embedding}")

# Step 5: Use the Chroma retriever with the text splitter and embeddings
retriever = db.as_retriever(
    search_kwargs={
        "k": 5  # Specify the number of results to return
    }
)

# Step 6: Retrieve relevant documents based on the query
try:
    relevant_docs_with_scores = retriever.invoke(query)

    # Display the relevant results with metadata and scores
    print("\n--- Relevant Documents with Scores ---")
    if relevant_docs_with_scores:
        for i, doc in enumerate(relevant_docs_with_scores, 1):
            # Printing the document content and metadata
            print(f"Document {i}:\n{doc.page_content}\n")
            if 'source' in doc.metadata:
                print(f"Source: {doc.metadata['source']}\n")
            # Check if score exists in metadata and print it
            if 'score' in doc.metadata:
                print(f"Score: {doc.metadata['score']}\n")
    else:
        print("No relevant documents retrieved.")
except Exception as e:
    print(f"An error occurred during retrieval: {e}")
