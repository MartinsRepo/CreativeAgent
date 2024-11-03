# %%
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import pickle
import json
import vosk
import pyaudio
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# %%
def initialisation():
    # Language in which you want to convert
    language = 'en'
    
    # Load the LLaMA 3.2 model from Ollama backend
    model = OllamaLLM(model='llama3.2')
    
    # Load the Vosk model
    #modelvoice = vosk.Model(lang="en-us")    
    #recognizer = vosk.KaldiRecognizer(model, 16000 )
    
    # Set up long-term memory using ChromaDB
    memory = ConversationBufferMemory()
    
    # Set up Chroma vector store
    #vectorstore = Chroma(persist_directory='./chroma_vectorstore', embedding_function=model.embed_text)
    from langchain_community.embeddings.ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model='llama3.2')
    vectorstore = Chroma(persist_directory='./chroma_vectorstore', embedding_function=embeddings)
    
    
    
    # Set up retrieval chain for Chroma vectorstore
    from langchain.chains import RetrievalQA
    retriever = vectorstore.as_retriever()
    retriever = vectorstore.as_retriever(search_type='similarity')
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    
    return model, language, memory, retrieval_chain

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
def create_tools(retrieval_chain, model, memory):
    # Set up search and Wikipedia tools
    duckduckgo = DuckDuckGoSearchResults()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # Create a custom tool for the retrieval chain
    from langchain.tools import Tool
    retrieval_tool = Tool(
        name="ChromaRetrieval",
        func=lambda q: retrieval_chain.run(q),
        description="Use this tool to retrieve information from Chroma vectorstore"
    )
    
    # Create the React agent with memory, internet search, and Wikipedia access
    agent = initialize_agent(
        tools=[retrieval_tool, wikipedia, duckduckgo],
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        allowed_tools=["ChromaRetrieval", "wikipedia", "duckduckgo"]
    )
    
    return retrieval_tool, agent

# %%
# Start a conversation with the agent
def start_conversation(agent, memory):
    
    assistant_intro = """You are an AI assistant specialized in artificial intelligence, robotics, and automotive technology. \
    Your job is to help generate maximum of three creative ideas and answer questions on these topics. Use the latest research and trends in your responses.\nExplain exactly three ideas and make your answer short. Stop after the answer.\n\nQuestion: """

    print("Agent: How can I help you? Type 'exit' to end the conversation.")
    conversation_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            # Before exiting the program
            with open('conversation_history.pkl', 'wb') as f:
                pickle.dump(memory, f)
            break
        
        # Modify prompt and add user input
        modified_prompt = assistant_intro + user_input

        # Agent processes user input
        response = agent.invoke(modified_prompt)

        # Print agent's response
        print(f"Agent: {response}")


# %%
if __name__ == "__main__":
    # start initialisation
    model, language, memory, retrieval_chain = initialisation()
    
    device, stream = init_microphon()
    
    # Create the custom tool
    retrieval_tool, agent = create_tools(retrieval_chain, model, memory)
    
    start_conversation(agent, memory)
    


