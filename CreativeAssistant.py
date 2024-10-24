#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:09:50 2024

@author: martin
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchResults
import requests
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play

# Language in which you want to convert
language = 'en'

# Load the Ollama LLM (LLaMA 3)
ollama_llm = OllamaLLM(model="llama3.2")

# DuckDuckGo Search for internet access

def duckduckgo_search(query: str) -> str:
    search_tool = DuckDuckGoSearchResults()
    try:
        # Disable SSL certificate verification
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        results = search_tool.invoke(query)
        return results
    except Exception as e:
        return f"Error during search: {str(e)}" 
    
prompt_template = """You are an AI assistant specialized in artificial intelligence, robotics, and automotive technology. 
Your job is to help generate creative ideas and answer questions on these topics. Use the latest research and trends in your responses.
Explain exactly three ideas and make your answer short. Stop after the answer.

Question: {question}

Assistant: """
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

# Create the LLM chain for creative question answering
creative_chain = LLMChain(llm=ollama_llm, prompt=prompt)

# Create tools for the agent
tools = [
    Tool(
        name="CreativeAssistant",
        func=creative_chain.run,
        description="Generate creative ideas and answers related to AI, robotics, and autonomous vehicles."
    ),
    Tool(
        name="DuckDuckGoSearch",
        func=duckduckgo_search,
        description="Search the web for the latest information."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=ollama_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example queries
queries = [
    "What are some creative ideas for AI-driven automotive agents?",
    "How can robotics improve autonomous vehicle systems?",
    "What are the latest trends in AI for automotive innovation?"
]

response = ""
for query in queries:
    #print(f"Query: {query}")
    text = agent.run(query)
    response = response + text

    # Creating the gTTS object
    speech = gTTS(text=text, lang=language, slow=False)

    audio_buffer = io.BytesIO()
    speech.write_to_fp(audio_buffer)

    # Convert to AudioSegment and play
    audio_buffer.seek(0)

    audio = AudioSegment.from_file(audio_buffer, format="mp3")
    play(audio)

# Print console output
print(f"Response: {response}\n")