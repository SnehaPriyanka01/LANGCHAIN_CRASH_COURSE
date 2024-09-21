from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama

#load environment variables from .env
load_dotenv()

print("load env")
#create a chatollama model
model = ChatOllama(base_url='https://ollama.dealwallet.com/',model = 'llama3')
print ("llama3 model created")
#invoke the model with a message
result = model.invoke("what is 81 divided by 9?")
print("full result")
print(result)
print("contentonly")
print(result.content)