from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama  # For chat interactions

# Load environment variables from .env file
load_dotenv()

# Define your tools and functions
def get_current_time(*args, **kwargs):
    """Returns the current time."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches Wikipedia."""
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that. Error: {str(e)}"

# Define Tools
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize the ChatOllama model
llm = ChatOllama(model="llama3", base_url="https://ollama.dealwallet.com/")

# Create a structured Chat Agent
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor for managing interactions
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Initial system message
initial_message = "You are an AI assistant that can provide helpful answers using available tools."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add user message to memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with user input
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add agent response to memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
