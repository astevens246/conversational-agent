# Import required libraries
# %pip install -q langchain langchain_experimental openai python-dotenv langchain_openai
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000, temperature=0)

# Create a simple in-memory store for chat histories
store = {}


def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define response formatting options
formatting_options = {
    "default": "",
    "eli5": "Explain like I'm 5.",
    "bullets": "Use bullet points.",
    "rhyme": "Make it rhyme.",
}

# Function to format the response based on user input
def format_input(user_input, format_type="default"):
    if format_type in formatting_options:
        return f"{formatting_options[format_type]} {user_input}"
    else:
        return user_input

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
print("AI (default format):", response1.content)

# Second interaction
formatted_input = format_input("Explain quantum physics.", "eli5")
response2 = chain_with_history.invoke(
    {"input": formatted_input},
    config={"configurable": {"session_id": session_id}},
)
print("\nAI (ELI5):", response2.content)

# Third interaction
formatted_input = format_input("What is the history of the United States?", "bullets")
response3 = chain_with_history.invoke(
    {"input": formatted_input},
    config={"configurable": {"session_id": session_id}},
)
print("\nAI (Bullets):", response3.content)

# Fourth interaction
formatted_input = format_input("Explain the rules of golf.", "rhyme")
response4 = chain_with_history.invoke(
    {"input": formatted_input},
    config={"configurable": {"session_id": session_id}},
)
print("\nAI (Rhyme):", response4.content)
# # Uncomment the following lines to print the conversation history

# # Print the conversation history
# print("\nConversation History:")
# for message in store[session_id].messages:
#     print(f"{message.type}: {message.content}")
