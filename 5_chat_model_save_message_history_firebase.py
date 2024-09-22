import firebase_admin
from firebase_admin import credentials, firestore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
import datetime

# Initialize Firebase Admin SDK
# Update the path to match the location of the credentials file in your project
cred = credentials.Certificate(r'C:\Users\sneha\Downloads\chat-history-project-firebase-adminsdk-8b59a-7d2dc342ff.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to save message to Firebase Firestore
def save_message_to_firebase(user_message, ai_response):
    try:
        doc_ref = db.collection("chat_history").document()  # Create a new document in the "chat_history" collection
        doc_ref.set({
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": datetime.datetime.now()  # Add timestamp to track when the conversation occurred
        })
        print("Message saved to Firestore successfully.")
    except Exception as e:
        print(f"An error occurred while saving to Firestore: {e}")

# Define the question prompt template
question_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

# Initialize the Ollama model
# No need for dotenv, just pass the base URL and model directly
model = ChatOllama(base_url='https://ollama.dealwallet.com', model='llama3')

# Create the LLMChain with the prompt and the Ollama model
question_chain = LLMChain(llm=model, prompt=question_prompt)

def main():
    print("Welcome to the AI conversation. Type 'exit' to end the conversation.")
    while True:
        # Get user's question/input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Run the question through the chain to get the AI response
        try:
            response = question_chain.run(question=user_input)
            # Print the AI response
            print(f"AI: {response}")
            # Save the conversation (user message and AI response) to Firebase
            save_message_to_firebase(user_input, response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
