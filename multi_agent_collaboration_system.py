import os
import time

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)

class Agent:
    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills
        self.llm = llm

    def process(self, task: str, context: List[Dict] = None) -> str:
        messages = [
            SystemMessage(content=f"You are {self.name}, a {self.role}. Your skills include: {', '.join(self.skills)}. Respond to the task based on your role and skills.")
        ]
        
        if context:
            for msg in context:
                if msg['role'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))
        
        messages.append(HumanMessage(content=task))
        response = self.llm.invoke(messages)
        return response.content