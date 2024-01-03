from dotenv import load_dotenv
import pandas as pd
import os
import time
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


directory = 'D:\Projects\Capstone\Chat-CSEC'
os.chdir(directory)

result = load_dotenv('keys.env', verbose=True)

if result:
    print("environment file specified")
else:
    print("you do not have an env file with the api key. this wont work")

OPENAI_API_KEY = os.getenv('key')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# workaraound to have relevance scores. Inspired by top solution from: https://github.com/langchain-ai/langchain/issues/5067
class MyVectorStoreRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs

meta_list = ['none', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA', 'MITRE']

vectordb = Chroma(persist_directory="database", embedding_function=embeddings)
model_name = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)
prompt_template = """"You are a malware expert that answers in very high detail. Get every technical detail and explain as if you are talking to an expert. 
{context}
{chat_history}
Question: {question}
Answer: 
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context", "chat_history"])
chain_type_kywargs2 = {"prompt": PROMPT}
memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key="answer")

def general_search(query):
    scoring = ""
    tester = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory,combine_docs_chain_kwargs={"prompt": PROMPT},max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    stuffed = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, chain_type="stuff",combine_docs_chain_kwargs={"prompt": PROMPT},max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    try:
        result = tester({"question": query}, return_only_outputs=True)
        print()
        documents = result["source_documents"]
        position = 1
        for res in documents:
            print(f"document # {position}:")
            print(f"distance is: {res.metadata['score']}")
            print(f"source = {res.metadata['source']}")
            print(f"topic = {res.metadata['topic']}")
            print(res.page_content)
            print()
            print()
            print()
            position += 1
        score = 0
        for res in documents:
            value = res.metadata['score']
            score += value
        final_score = score/(len(documents))
        print(f"average score is: {final_score}")
        print()
        print(f"rephrased question: {result['generated_question']}")
        print()
        print("answer:")
        print(result["answer"])
        print()
        memory.clear()
        scoring = [query, final_score, result['answer']]
        time.sleep(10)
        return scoring

    except Exception as E:
        try:
            result = stuffed({"question": query}, return_only_outputs=True)
            print()
            documents = result["source_documents"]
            position = 1
            for res in documents:
                print(f"document # {position}:")
                print(f"distance is: {res.metadata['score']}")
                print(f"source = {res.metadata['source']}")
                print(f"topic = {res.metadata['topic']}")
                print(res.page_content)
                print()
                print()
                print()
                position += 1
            score = 0
            for res in documents:
                value = res.metadata['score']
                score += value
            final_score = score/(len(documents))
            print(f"average score is: {final_score}")
            print()
            print(f"rephrased question: {result['generated_question']}")
            print()
            print("answer:")
            print(result["answer"])
            print()
            memory.clear()
            scoring = [query, final_score, result['answer']]
            time.sleep(10)
            return scoring
        except Exception as E:
            print(str(E))
            scoring = [query, "NA", "NA"]
            time.sleep(10)
            return scoring
    
                
def filter_search(query, filter):
    scoring = []
    tester = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={"filter": {'topic' : str(filter)},'k': 15}), memory=memory,combine_docs_chain_kwargs={"prompt": PROMPT},max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    stuffed = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={"filter": {'topic' : str(filter)},'k': 15}), memory=memory, chain_type="stuff", combine_docs_chain_kwargs={"prompt": PROMPT},max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    try:
        result = tester({"question": query}, return_only_outputs=True)
        print()
        documents = result["source_documents"]
        position = 1
        for res in documents:
            print(f"document # {position}:")
            print(f"distance is: {res.metadata['score']}")
            print(f"source = {res.metadata['source']}")
            print(f"topic = {res.metadata['topic']}")
            print(res.page_content)
            print()
            print()
            print()
            position += 1
        score = 0
        for res in documents:
            value = res.metadata['score']
            score += value
        final_score = score/(len(documents))
        print(f"average score is: {final_score}")
        print()
        print(f"rephrased question: {result['generated_question']}")
        print()
        print("answer:")
        print(result["answer"])
        print()
        memory.clear()
        scoring = [query, final_score, result['answer']]
        time.sleep(10)
        return scoring
    except Exception as E:
        try:
            result = stuffed({"question": query}, return_only_outputs=True)
            print()
            documents = result["source_documents"]
            position = 1
            for res in documents:
                print(f"document # {position}:")
                print(f"distance is: {res.metadata['score']}")
                print(f"source = {res.metadata['source']}")
                print(f"topic = {res.metadata['topic']}")
                print(res.page_content)
                print()
                print()
                print()
                position += 1
            score = 0
            for res in documents:
                value = res.metadata['score']
                score += value
            final_score = score/(len(documents))
            print(f"average score is: {final_score}")
            print()
            print(f"rephrased question: {result['generated_question']}")
            print()
            print("answer:")
            print(result["answer"])
            print()
            memory.clear()
            scoring = [query, final_score, result['answer']]
            time.sleep(10)
            return scoring
        except Exception as E:
            print(str(E))
            scoring = [query, "NA", "NA"]
            time.sleep(10)
            return scoring

def virus_specific(query, filter):
    ranking = []
    questions = [f"what is {query}",
                 f"who made {query}",
                 f"what are the zero days that {query} took advantage of",
                 f"Give more information on the zero day vulnerabilities that {query} took advantage of.",
                 f"Tell the story of how {query} was carried out.",
                 f"when was {query} discovered",
                 f"what were the commands that {query} used",
                 f"what are the yara rules associated with {query}",
                 f"give more information about yara rules associated with {query} and write a yara rule for {query}",
                 f"What were the dlls tha {query} use",
                 f"how did {query} use its dlls",
                 f"list all the relevant files that {query} used",
                 f"list the tactics and procesdure that {query} used",
                 f"explain the cyberkill chain of {query}",
                 f"Explain every dll and how {query} used them in the environment.",
                 f"Did {query} have the ability to exfiltrate and cover its tracks?",
                 f"How did {query} cover its tracks? What were the relevant components that achieved this?",
                 f"Did {query} have the ability to achieve persistence?",
                 f"What kind of malware was {query}? Worm, trojan, rootkit, etc..",
                 f"What would {query} fall under a different category?",
                 f"Explain {query} components that relate it to a trojan, worm, or other type of malware.",
                 f"Did {query} have any cryptographic components?",
                 f"Did {query} use C&Cs? if so then what were they named and how were they used?",
                 f"what operating systems did {query} run on?",
                 f"Write the psedocode with the dlls mentioned that {query} would have used to achieve its purpose.",
                 f"Are there other specimens that are similar to {query}?",
                 f"How would industries protec themselves in the future based on that was learned from the {query} attack?",
                 f"List all CVEs you can that are related to {query}."
                 ]
    if filter == 'none':
        for q in questions:
            rank = general_search(q)
            ranking.append(rank)
    else:
        for q in questions:
            rank = filter_search(q, filter)
            ranking.append(rank)
    print('--------------------------------------------------------')
    df = pd.DataFrame(ranking, columns=['Query', 'Relevance Score', 'Answer'])
    csv_name = f"{query}_test.csv" 
    df.to_csv(csv_name, index=False)
    df.style.set_properties(subset=df.columns, **{'white-space':'pre-wrap'})
    print("check test excel file for results")

def malware_dll():
    return None

def malware_components():
    return None

def malware_assembly():
    return None

def malware_forensics():
    return None

def malware_toolkits():
    return None

if __name__ == "__main__":
    topic = str(input("what kind of virus are you asking about?\n"))
    while True:
        filt = str(input(f"select a topix from {meta_list}. if you don't want filtering, type none.\n"))
        if filt in meta_list:
            break
        else:
            print()
            print("enter a valid option.")
            print()
    
    virus_specific(topic, filt)
