# Import necessary libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_ollama.chat_models import ChatOllama

# Step 1: Initialize the LLM (ChatOllama)
ollama_model = ChatOllama(base_url='https://ollama.dealwallet.com', model='llama3')

# Step 2: Create multiple prompt templates
# Prompt 1: Initial question
initial_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Please provide a detailed response to the following question: {question}"
)

# Prompt 2: Follow-up based on previous response
followup_prompt_template = PromptTemplate(
    input_variables=["response"],
    template="Based on the previous response, can you elaborate more on this: {response}?"
)

# Step 3: Define the chains
# First Chain: Ask the initial question and generate a response
initial_chain = LLMChain(llm=ollama_model, prompt=initial_prompt_template)

# Second Chain: Take the response from the first chain and ask for further elaboration
followup_chain = LLMChain(llm=ollama_model, prompt=followup_prompt_template)

# Step 4: Combine the chains into a sequential flow
extended_chain = SimpleSequentialChain(chains=[initial_chain, followup_chain])

# Step 5: Main logic to run the extended chain
def main():
    print("Welcome to the Extended Chain example.")

    # Ask for the user's question
    user_question = input("Enter your question: ")

    # Step 6: Run the extended chain and get the final response
    final_response = extended_chain.invoke(user_question)

    # Print the AI's final response
    print(f"\nAI Final Response:\n{final_response}")

if __name__ == "__main__":
    main()
