from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document
from typing import List

directory = ''
os.chdir(directory)

result = load_dotenv('keys.env', verbose=True)

if result:
    print("environment file specified")
else:
    print("you do not have an env file with the api key. this wont work")

OPENAI_API_KEY = os.getenv('key')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


meta_list = ['', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA']

vectordb = Chroma(persist_directory="database", embedding_function=embeddings)
model_name = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="database", embedding_function=embeddings)
while True:
    test_type = input(str("would you like to do the general test or the filtered test? answer general or filter\n"))
    if test_type == "general" or test_type == "filter":
        break
    else:
        print()
        print("please answer 'general' or 'filter'. no capital letters")
        print()
if test_type == "general":
    print("starting general search test.")
    print()
    print("type exit to finish the program")
    print()
    while True:
        query = input(str("Human:"))
        if query == "exit":
            break
        else:
            result = vectorstore.similarity_search_with_score(query=query, k = 15)
            for res in result:
                print(f"document # {result.index(res)}:")
                print(f"distance is: {res[1]}")
                print(f"source = {res[0].metadata['source']}")
                print(f"topic = {res[0].metadata['topic']}")
                print(res)
                print()
                print()
                print()
            score = 0
            for res in result:
                #distance is res[1]
                score += res[1]
            final_score = score/(len(result))
            print(f"average score is: {final_score}")
else:
    print("starting filter search test.")
    while True:
        print(meta_list)
        topic_filter = input(str("type a topic from the list above\n"))
        if topic_filter in meta_list:
            break
        else:
            print()
            print("please type a topic from the list.")
            print()
    print("type exit to finish the program")
    print()
    while True:
        query = input(str("Human:"))
        if query == "exit":
            break
        else:
            result = vectorstore.similarity_search_with_score(query=query, k = 15, filter={'topic':str(topic_filter)})
            for res in result:
                print(f"document # {result.index(res)}:")
                print(f"distance is: {res[1]}")
                print(f"source = {res[0].metadata['source']}")
                print(f"topic = {res[0].metadata['topic']}")
                print(res)
                print()
                print()
            print()
            score = 0
            for res in result:
                #distance is res[1]
                score += res[1]
            final_score = score/(len(result))
            print(f"average score is: {final_score}")
                