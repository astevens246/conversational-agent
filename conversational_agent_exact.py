# Import required libraries
# %pip install -q langchain langchain_experimental openai python-dotenv langchain_openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load environment variables and initialize the language model
load_dotenv()
llm = ChatOpenAI(model="gpt-4", max_tokens=1000, temperature=0)

# Create a simple in-memory store for chat histories
store = {}


def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Combine the prompt and model into a runnable chain
chain = prompt | llm

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain, get_chat_history, input_messages_key="input", history_messages_key="history"
)
