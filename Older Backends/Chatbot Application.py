from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# the bottom is the libraries for the user interface
import streamlit as st
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
# change the directory to the one your python file is currently in.

st.set_page_config(page_title="Chat CSEC: Malware Chatbot")
st.title("Chat CSEC: Malware Chatbot")

# streaming handler inspired by one of langchains github guides for streamlit 

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
    
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # rephrase question has its own prompt. its tokens will be ignored in streaming only. 
        if prompts[0].startswith("Human: Given the following conversation"):
            self.run_id_ignore_token = kwargs.get("run_id")
    

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

# directory full path
directory = ''
os.chdir(directory)

result = load_dotenv('keys.env', verbose=True)

if result:
    print("environment file specified")
else:
    print("you do not have an env file with the api key. this wont work")

OPENAI_API_KEY = os.getenv('key')


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

model_name = 'gpt-3.5-turbo'
#model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

msg = StreamlitChatMessageHistory()

memory=ConversationBufferMemory(memory_key="chat_history",chat_memory= msg, return_messages=True)


Malware_db = Chroma(persist_directory='database', embedding_function=embeddings)


# regular prompt template

prompt_template = """You are a malware expert that answers in very high detail. 
{context}
{chat_history}
Question: {question}
Answer: 
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context", "chat_history"])
chain_type_kywargs2 = {"prompt": PROMPT}

# test prompt template for exception

prompt_template2 = """You are a malware expert that answers in very high detail. Get every technical detail and explain as if you are talking to an expert. 
{context}
{chat_history}
Question: {question}
Answer: 
"""

PROMPT2 = PromptTemplate(template=prompt_template2, input_variables=["question", "context", "chat_history"])
chain_type_kywargs3 = {"prompt": PROMPT2}

# Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={"filter": metadata,'K': 50}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT})
model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True)

#depth = st.selectbox('this is how in depth the knowlegebase used should be. select high for a higher level overview, or deep for in depth.', ('high', 'deep'))
topic = st.selectbox('What topic of malware would you like to specificalyl search for? you can choose general for all purpose (recommended). IDC will show lesser known IDCs as most are oficially classified as worms or trojans. havex, black energy, industroyers are trojans while stuxnet, and duqu are worms.', ('general', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA'))


#if str(depth) == 'high':
#    chosen_prompt = PROMPT
#elif str(depth) == 'deep':
#    chosen_prompt = PROMPT2

if str(topic) == 'general':
    Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, verbose=True, max_tokens_limit = 4090)
    Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={'k': 10}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, verbose=True, chain_type="stuff", max_tokens_limit = 4090)
else:
    filter = {'topic': str(topic) }
    Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={"filter": filter,'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, verbose=True, max_tokens_limit = 4090)
    Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={"filter": filter,'k': 10}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, verbose=True, chain_type="stuff", max_tokens_limit = 4090)

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content = "Lets talk malware")]

for message in st.session_state.messages:
    st.chat_message(message.role).write(message.content)

if prompt := st.chat_input("say something"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            stream_handler = StreamHandler(st.empty())
            response = Malware_context.run(prompt, callbacks=[stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))   
        except Exception as E:
            stream_handler = StreamHandler(st.empty())
            response = Malware_exception.run(prompt, callbacks=[stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response)) 



