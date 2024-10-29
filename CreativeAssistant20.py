# %%
'''
################################################################
# File: talkollama.py
# Purpose:
# - voice detection via microphone
# - Q/A to a local Large Language Model
# - Voice input
# - Voice output
# - no internet access necessary
#
# Licence: Apache 2.0
#
# Martin Hummel
# created on Thu Oct 24 09:09:50 2024
# v2.0
################################################################
'''

# %%
import io
import json
import pyaudio
import vosk
import requests

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchResults

from gtts import gTTS

from pydub import AudioSegment
from pydub.playback import play

# Language in which you want to convert
language = 'en'

# %%
# Initialize the Ollama LLM (LLaMA 3)
ollama_llm = OllamaLLM(model="llama3.2")

# Load the Vosk model
model = vosk.Model(lang="en-us")    
recognizer = vosk.KaldiRecognizer(model, 16000 )

# %%
def duckduckgo_search(query: str) -> str:
            search_tool = DuckDuckGoSearchResults()
            try:
                # Disable SSL certificate verification
                print('###',query)
                requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
                results = search_tool.invoke(query)
                print('++++',results)
                return results
            except Exception as e:
                return f"Error during search: {str(e)}"

# %%
# Create the prompt template
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

# %%
# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=ollama_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    action_input_format="Action Input:"
)

# %%
def handle_query(query: str):
    # Run the agent with the query
    print('>>>>>',query)
    response = agent.run(query)
    print('<<<<<',response)

    # Creating the gTTS object
    speech = gTTS(text=response, lang='en', slow=False)
    print('+++++',speech)
    audio_buffer = io.BytesIO()
    speech.write_to_fp(audio_buffer)

    # Convert to AudioSegment and play
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format="mp3")
    play(audio)

    return response

# %%
def init_microphon():
    # Start audio stream with error handling for device selection
    # Open the microphone stream
    device = pyaudio.PyAudio()
    stream = device.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=16000,
                         input=True,
                         frames_per_buffer=8192
                         )
        
    return device, stream

# %%
def start_listening(device, stream):
    
    print("Listening...")
    
    # Specify the path for the output text file
    output_file_path = "recognized_text.txt"
    
    def stop_listening(device, stream): 
        print("Voice Assistant Stopped.")    
        stream.stop_stream()
        stream.close()
        device.terminate()

    # Open a text file in write mode using a 'with' block
    #try:
    with open(output_file_path, "w") as output_file:
        print("Listening for speech. Say 'Terminate' to stop.")
        # Start streaming and recognize speech
        while True:
                data = stream.read(1024)#read in chunks of 4096 bytes
                if recognizer.AcceptWaveform(data):#accept waveform of input voice
                    
                    # Parse the JSON result and get the recognized text
                    result = json.loads(recognizer.Result())
                    recognized_text = result['text']
                    
                    # Write recognized text to the file
                    output_file.write(recognized_text + "\n")
                    print(recognized_text)
                    response = ""
                    response = handle_query(recognized_text)
                    #print(response)
                    
                    # Check for the termination keyword
                    if "terminate" in recognized_text.lower():
                        print("Termination keyword detected. Stopping...")
                        # Stop and close the stream
                        stop_listening(device, stream)
                        break
        
    #except Exception as e:
    #    print("KeyboardInterrupt: Stopping the Voice Assistant...")
    #    stop_listening(device, stream)

# %%
if __name__ == "__main__":
    device, stream = init_microphon()
    start_listening(device, stream)
    
    


