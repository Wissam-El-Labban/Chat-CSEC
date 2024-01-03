import os
import getpass
import re
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.agents import initialize_agent, Tool, AgentOutputParser, AgentExecutor, LLMSingleActionAgent, ZeroShotAgent, AgentType
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import List, Union
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
PINECONE_API_KEY = keys.PINECONE_API_KEY
PINECONE_API_ENV = keys.PINECONE_API_ENV



embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

model_name = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

memory=ConversationBufferWindowMemory(k=2)
Malware_db = Chroma(persist_directory='Malware', embedding_function=embeddings)
Malware_csv = Chroma(persist_directory='Malware_csv', embedding_function=embeddings)



Malware_context = RetrievalQA.from_chain_type(llm=model, retriever=Malware_db.as_retriever(search_type="similarity", search_kwargs={'K': 10}), chain_type="stuff")
Malware_csv= RetrievalQA.from_chain_type(llm=model, retriever=Malware_csv.as_retriever(search_type="similarity", search_kwargs={'K': 10}), chain_type="stuff")

Tools = [
    Tool(name= "Malware_context_store", func=Malware_context.run, description="more definitive context on malware that will be provided while building the answer"),
    Tool(name= "Malware_csv_store", func=Malware_csv.run, description="more definitive context on malware that will be provided while building the answer"),
]


prefix = """You are a malware expert. Answering the following questions as best you can. You have access to the following tools:
Malware_context_store: this contains general informatoin about malware
Malware_csv_store: this contains supplementary data that should be in the final answer if any meaningful information is found from it. 
"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    Tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=model, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=Tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=Tools, verbose=True, memory=memory,
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
