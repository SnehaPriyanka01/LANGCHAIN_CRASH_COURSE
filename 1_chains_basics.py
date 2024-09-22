from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()
ollama_model = ChatOllama(
    base_url='https://ollama.dealwallet.com',
    model='llama3'
)

# Step 2: Create a PromptTemplate
# This is where you define the structure of the prompt.
prompt_template = PromptTemplate(
    input_variables=["question"],  # Define input variables to replace
    template="You are a helpful assistant. Please answer the following question: {question}"
)

# Step 3: Set up the LLMChain
# The LLMChain combines the prompt template and the LLM to generate responses
llm_chain = LLMChain(
    llm=ollama_model,  # Use the ChatOllama model
    prompt=prompt_template  # Use the defined prompt template
)

# Step 4: Define a function to use the chain and get responses
def ask_question(question):
    # This method runs the LLMChain and returns the response from the AI
    return llm_chain.run(question=question)

# Step 5: Example Usage
if __name__ == "__main__":
    # Ask a question to the chain and print the response
    user_question = input("What is your question? ")
    response = ask_question(user_question)

    print(f"AI Response: {response}")
