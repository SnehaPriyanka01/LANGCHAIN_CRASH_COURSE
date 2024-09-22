from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(base_url='https://ollama.dealwallet.com/', model='llama3')

# Initialize conversation history
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

print("Real-Time Chat with Ollama AI")
print("Type 'exit' to end the conversation.\n")

# Real-time conversation loop
while True:
    # Take user input
    user_input = input("You: ")

    # End the conversation if user types 'exit'
    if user_input.lower() == 'exit':
        print("Ending the conversation. Goodbye!")
        break

    # Add the user's message to the conversation history
    messages.append(HumanMessage(content=user_input))

    # Invoke the model with the current conversation history
    result = model.invoke(messages)

    # Display the AI's response
    print(f"AI: {result.content}")

    # Add the AI's response to the conversation history
    messages.append(AIMessage(content=result.content))
