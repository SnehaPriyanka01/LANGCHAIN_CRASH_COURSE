from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_ollama.chat_models import ChatOllama  # Importing ChatOllama for Llama3 chat model
from langchain_core.tools import Tool

# Load environment variables from .env file
load_dotenv()

# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time
    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        description="Useful for when you need to know the current time",  # Description of the tool
    ),
]

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOllama model using ollama.dealwallet.com
llm = ChatOllama(
    model="llama3",  # Specify the model you want to use
    temperature=0,
    base_url="https://ollama.dealwallet.com",  # Set the base URL for Ollama
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)
