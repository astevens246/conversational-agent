# Import required libraries
# %pip install -q langchain langchain_experimental openai python-dotenv langchain_openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load environment variables and initialize the language model

llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000, temperature=0)

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

# Example usage
session_id = "user_123"

# First interaction
response1 = chain_with_history.invoke(
    {"input": "Hello! How are you?"},
    config={"configurable": {"session_id": session_id}},
)
print("AI:", response1.content)

# Second interaction
response2 = chain_with_history.invoke(
    {"input": "What was my previous message?"},
    config={"configurable": {"session_id": session_id}},
)
print("AI:", response2.content)

# Print the conversation history
print("\nConversation History:")
for message in store[session_id].messages:
    print(f"{message.type}: {message.content}")
