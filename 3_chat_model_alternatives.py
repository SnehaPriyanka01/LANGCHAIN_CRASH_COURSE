from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(base_url='https://ollama.dealwallet.com/', model='llama3')
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from HumanMessage: {result.content}")

# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
#model = ChatAnthropic(model="claude-3-opus-20240229")

#result = model.invoke(messages)
#print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

#result = model.invoke(messages)
#print(f"Answer from Google: {result.content}")
