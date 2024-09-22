# Import necessary libraries
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

# Step 1: Initialize the LLM (ChatOllama)
ollama_model = ChatOllama(base_url='https://ollama.dealwallet.com', model='llama3')

# Step 2: Create a PromptTemplate
# This template defines how the question will be formatted when sent to the LLM
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are a wise assistant. Please provide a detailed response to the following question: {question}"
)

# Step 3: Manually construct a prompt using the template
def create_prompt(question):
    return prompt_template.format(question=question)

# Step 4: Manually run the LLM (ChatOllama) using the formatted prompt
def run_llm(prompt):
    # This method should use the correct call to ChatOllama
    return ollama_model.predict(prompt)  # Use predict to call the model

# Step 5: Main logic to run the chain "under the hood"
def main():
    print("Welcome to the 'Chains Under the Hood' example")

    # Ask for the user's question
    user_question = input("Enter your question: ")

    # Step 6: Create the prompt manually
    formatted_prompt = create_prompt(user_question)
    print(f"\nGenerated Prompt:\n{formatted_prompt}\n")

    # Step 7: Pass the prompt to the LLM manually and get the response
    response = run_llm(formatted_prompt)

    # Print the AI's response
    print(f"AI Response:\n{response}")  # Directly print the response

if __name__ == "__main__":
    main()
