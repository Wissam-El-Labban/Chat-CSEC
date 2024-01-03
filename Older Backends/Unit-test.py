import os
import getpass
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
# importing key variables from key file
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
PINECONE_API_KEY = keys.PINECONE_API_KEY
PINECONE_API_ENV = keys.PINECONE_API_ENV


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# temperature in the function below is 0. this is the randomness variable. a temperature value > 1 will be completely random. 
# the value below is at 0 to have no randomness. this coupled with our relevant document search is our method of mitigating hallucinations from the LLM

#llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0, openai_api_key=API_Key)

prompt_template = """Use the following pieces of retrieved documents to answer the question at the end. Please think rationally, take chat history into consideration, and answer from your own knowledge base. If you really can't construct an answer or the documents does not provide information about the specific query, then answer with 'no'.
{context}
{chat_history}
question: {question}
Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
chain_type_kywargs = {"prompt": PROMPT}


model_name = 'gpt-3.5-turbo'


query = ''

global memory

model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index= "test1" 
# put in the name of your pinecone index here

vectordb = Pinecone.from_existing_index(index_name=index, embedding=embeddings)
qa = ConversationalRetrievalChain.from_llm(llm=model, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 10}), memory=memory, combine_docs_chain_kwargs={"prompt":PROMPT})
gpt_qa = ConversationChain(llm=model)

# the below is our current chat backend. in later sprints we will have user interface. but the below is fine for our testing purposes
# if the user types exit, the loop ends
while True:
    query = str(input("Human: "))
    if query == 'exit':
        break
    print()
    response = qa.run(query)
    
    if response == 'no' or response == 'No' or response == 'no.' or response == 'No.':
        print("referring to chatgpt")
        gptresponse = gpt_qa.run(query)
        print(gptresponse)
        memory.save_context({"input": query}, {"output": gptresponse})
        print()
    else:
        print(f'Chat CSEC: {response}')
        print()