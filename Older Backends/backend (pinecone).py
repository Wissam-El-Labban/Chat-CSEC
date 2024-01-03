import os
import getpass
import re
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.agents import initialize_agent, Tool, AgentOutputParser, AgentExecutor, LLMSingleActionAgent, ZeroShotAgent, AgentType, load_tools
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.utilities import SerpAPIWrapper, searx_search
from typing import List, Union
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
PINECONE_API_KEY = keys.PINECONE_API_KEY
PINECONE_API_ENV = keys.PINECONE_API_ENV
SERPER_API_KEY = keys.SERP_API_KEY

loader = WebBaseLoader('https://www.ibm.com/topics/malware')
data = loader.load()
print(data)

search = SerpAPIWrapper(serpapi_api_key=SERPER_API_KEY)
#search2 = searx_search()
#search3 = Gool

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

model_name = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=model_name, temperature=0.5, openai_api_key=OPENAI_API_KEY)

memory=ConversationBufferWindowMemory(k=2)
model_name = 'gpt-3.5-turbo'

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index= "test1" 
# put in the name of your pinecone index here


Malware_db = Pinecone.from_existing_index(index_name=index, embedding=embeddings)
Malwaredb = RetrievalQA.from_chain_type(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={'k': 20}), chain_type="stuff")
GPT = ConversationChain(llm=model)

Tools = [
    Tool(name= "Malware_context_store", func=Malwaredb.run, description="more definitive context on malware that will be provided while building the answer"),
    Tool(name= "ChatGPT", func=GPT.run, description="the main chatgpt that can be used to answer questions if malware_context fails")
]

tool_names = ["Malware_context_store", "ChatGPT"]

prefix = """You are a malware expert. Answering the following questions as best you can. You have access to the following tools:
Malware_context_store: this contains general informationn about malware
GPT: this is what you refer to if you cannot build an answer from malware_context_store. 
"""
FORMAT_INSTRUCTIONS= '''User the following format:
Action: the first action to take, should be Malware_context_store. If it does not yield a result, try GPT in the next action. both are from[{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. I will mix all my observations into it. 
Final Answer: the final answer to the original input question"""
'''

suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    Tools,
    prefix=prefix,
    format_instructions=FORMAT_INSTRUCTIONS,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")


llm_chain = LLMChain(llm=model, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=Tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=Tools, verbose=True, memory=memory, handle_parsing_errors=True
)

#agent_chain.run("give an example of malware cves")



while True:
    query = str(input("Human: "))
    if query == 'exit':
        break
    print()
    response = agent_chain.run(query)
    print(f'Chat CSEC: {response}')
    print()
