from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import requests

# Load environment variables from .env
load_dotenv()

# Simulated API request for features (mock implementation)
def get_product_features_ollama(product_name):
    # Mock URL, replace with actual API endpoint
    url = f"https://ollama.dealwallet.com/product/features/{product_name}"

    # Mock response based on the product name
    # Replace this with actual API call if available
    mock_responses = {
        "MacBook Pro": ["Retina display", "M1 chip", "Long battery life"],
        "iPhone": ["5G connectivity", "A15 Bionic chip", "Super Retina XDR display"]
    }

    features = mock_responses.get(product_name, ["Error fetching features"])
    return features

# Define pros analysis step
def analyze_pros(features):
    if not features:
        return "No pros available because no features were found."
    pros = [f"{feature} is beneficial" for feature in features]
    return "\n".join(pros)

# Define cons analysis step
def analyze_cons(features):
    if not features:
        return "No cons available because no features were found."
    cons = [f"{feature} may have drawbacks" for feature in features]
    return "\n".join(cons)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Simplify branches with LCEL
pros_branch_chain = RunnableLambda(lambda x: analyze_pros(x))

cons_branch_chain = RunnableLambda(lambda x: analyze_cons(x))

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    RunnableLambda(lambda x: get_product_features_ollama(x["product_name"]))  # Fetch features from Ollama
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})  # Parallelize pros and cons analysis
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))  # Combine results
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)
