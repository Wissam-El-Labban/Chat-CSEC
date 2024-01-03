from dotenv import load_dotenv
import os
import json
import glob
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import hashlib



def upload_pdf(metadata):
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    loc = input(str("are you working with remote or local pdfs? type remote or local.\n"))
    while True:
        if loc == "remote" or loc == "local":
            break
        else:
            print()
            print("please type 'remote' or 'local' . no capitals")
            print()
    if loc == "local":
        pdf_files = glob.glob("*.pdf")
    else:    
        try:
            with open("pdf.txt", 'r') as file:
                pdf_files = {line.strip() for line in file}
        except FileNotFoundError:
            print("pdf.txt not found")

    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    if '' in pdf_files:
        pdf_files.remove('')
    added = set()
    not_added = set()
    error = set()
    for path in pdf_files:
        value = ''
        collision = False
        try:
            exist = 0
            loader = PyPDFLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':
            print(f"working on {path}")
            print()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique


            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)
            
            #preparing metadata
            for t in texts:
                m = t.metadata
                m["source"] = str(path)
                m["topic"] = str(metadata)

            try:
                test = vectorstore.get(ids = hashes)
                existing = test['ids']
            except Exception as E:
                print(f"{path} has Sha3 collision issues")
                collision = True
            if collision:
                not_added.add(path)
            else:
                
                if len(existing) == 0:
                    print(f"adding {path} to database. it was never inside it.")
                    docsearch = Chroma.from_texts([t.page_content for t in texts], embeddings, ids = hashes, persist_directory='database', metadatas=[t.metadata for t in texts])
                    docsearch.persist()
                    print()
                    print(f"added {path} to database")
                    added.add(path)
                # if it does exist, then it will ignore it altogether
                elif len(existing) == len(hashes):
                    print(f'{path} has been uploaded before. skipping it')
                    not_added.add(path)
                # if there are parts of the document that already exist, this will take them out and then attempt to add the document to the db
                else:
                    print(f"sanitizing {path} contents.")
                    print()
                    # take the existing ids, localte them in the hash arry. it will correspond perfectly with its document chunk
                    # it will get the position and remove it from the array alongside its corresponding text. 
                    removed = []
                    existing = [id for id in test if id in hashes]
                    for id in existing:
                        position = hashes.index(id)
                        hashes.remove(id)
                        texts.remove(texts[position]) 
                        removed.append(id)
                    # the below are preconditions to move on. it is preferable to avoid hashing again. 
                    if len(hashes) == len(texts) and removed == existing:
                        test = vectorstore.get(ids = hashes)
                        existing = test['ids']
                        if len(existing) == 0:
                            #preparing metadata
                            for t in texts:
                                m = t.metadata
                                m["source"] = str(path)
                                m["topic"] = str(metadata)

                            print(f"adding sanitized {path} to database.")
                            docsearch = Chroma.from_texts([t.page_content for t in texts], embeddings, ids = hashes, persist_directory='database', metadatas=[t.metadata for t in texts])
                            docsearch.persist()
                            print()
                            print(f"added sanitized {path} to database")
                            added.add(path)
                            print()
                        else:
                            print(f"error in sanitizing {path}. check the code")
                            not_added.add(path)
                    else:
                        print(f"error in sanitizing {path}. check the code")
                        not_added.add(path)
        else:
            print(f"error loading {path}")
            not_added.add(path)
            error.add(path)
    if len(added) == len(pdf_files):
        print("all files have been added")
        print()
    else:
        print(f"{not_added} were not added into the database. check the runtime.")
        print()
    if len(error) != 0:
        print(f"{error} could not be reached")


def upload_web(metadata):
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    try:
        with open("web.txt", 'r') as file:
            links = {line.strip() for line in file}
    except FileNotFoundError:
        print("web.txt not found")

    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    if '' in links:
        links.remove('')
    added = set()
    not_added = set()
    error = set()
    for path in links:
        value = ''
        collision = False
        try:
            exist = 0
            loader = WebBaseLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique

            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)
            
            #preparing metadata
            for t in texts:
                m = t.metadata
                m["source"] = str(path)
                m["topic"] = str(metadata)

            try:
                test = vectorstore.get(ids = hashes)
                existing = test['ids']
            except Exception as E:
                print(f"{path} has Sha3 collision issues")
                collision = True
            if collision:
                not_added.add(path)
            else:
                if len(existing) == 0:
                    print(f"adding {path} to database. it was never inside it.")
                    docsearch = Chroma.from_texts([t.page_content for t in texts], embeddings, ids = hashes, persist_directory='database', metadatas = [t.metadata for t in texts])
                    docsearch.persist()
                    print()
                    print(f"added {path} to database")
                    added.add(path)
                # if it does exist, then it will ignor it altogether
                elif len(existing) == len(hashes):
                    print(f'{path} has been uploaded before. skipping it')
                    not_added.add(path)
                # if there are parts of the document that already exist, this will take them out and then attempt to add the document to the db
                else:
                    print(f"sanitizing {path} contents.")
                    print()
                    # take the existing ids, localte them in the hash arry. it will correspond perfectly with its document chunk
                    # it will get the position and remove it from the array alongside its corresponding text. 
                    removed = []
                    existing = [id for id in test if id in hashes]
                    for id in existing:
                        position = hashes.index(id)
                        hashes.remove(id)
                        texts.remove(texts[position]) 
                        removed.append(id)
                    # the below are preconditions to move on. it is preferable to avoid hashing again. 
                    if len(hashes) == len(texts) and removed == existing:
                        test = vectorstore.get(ids = hashes)
                        existing = test['ids']
                        if len(existing) == 0:
                            #preparing metadata
                            for t in texts:
                                m = t.metadata
                                m["source"] = str(path)
                                m["topic"] = str(metadata)

                            print(f"adding sanitized {path} to database.")
                            docsearch = Chroma.from_texts([t.page_content for t in texts], embeddings, ids = hashes, persist_directory='database', metadatas=[t.metadata for t in texts])
                            docsearch.persist()
                            print()
                            print(f"added sanitized {path} to database")
                            added.add(path)
                            print()
                        else:
                            print(f"error in sanitizing {path}. check the code")
                            not_added.add(path)
                    else:
                        print(f"error in sanitizing {path}. check the code")
                        not_added.add(path)
        else:
            print(f"error loading {path}")
            not_added.add(path)
            error.add(path)
    if len(added) == len(links):
        print("all links have been added")
        print()
    else:
        print(f"{not_added} were not added into the database. check the runtime.")
        print()
    if len(error) != 0:
        print(f"{error} are not reachable")

def delete_pdf():
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    loc = input(str("are you working with remote or local pdfs? type remote or local.\n"))
    while True:
        if loc == "remote" or loc == "local":
            break
        else:
            print()
            print("please type 'remote' or 'local' . no capitals")
            print()
    if loc == "local":
        pdf_files = glob.glob("*.pdf")
    else:    
        try:
            with open("pdf.txt", 'r') as file:
                pdf_files = {line.strip() for line in file}
        except FileNotFoundError:
            print("pdf.txt not found")
    # the below will load each pdf and split into mini documents for embedding
    if '' in pdf_files:
        pdf_files.remove('')
    deleted = set()
    not_deleted = set()
    error = set()
    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    for path in pdf_files:
        value = ''
        try:
            exist = 0
            loader = PyPDFLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique

            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db. skipping its deletion")
                not_deleted.add(path)
                print()
            elif len(existing) == len(hashes):
                print(f"{path} is all inside the database. deleting it in its entirety")
                vectorstore.delete(hashes)
                deleted.add(path)
            else:
                print()
                print(f"{path} partially exists in the db. deleteing its parts that exist")
                print()
                for id in existing:
                    if id not in hashes:
                        position = hashes.index(id)
                        hashes.remove(id)
                        texts.remove(texts[position])
                if len(hashes) == len(existing) and len(hashes) == len(texts):
                    vectorstore.delete(hashes)
                    deleted.add(path)
                else:
                    print(f"error in deleting {path}")
                    not_deleted.add(path)
                    print()
        else:
            print(f"{path} is unreachable")
            error.add(path)
            print()
    if len(deleted) == len(pdf_files):
        print("all pdfs have been deleted")
        print()
    else:
        print(f"{not_deleted} were not delted from the database. check the runtime.")
        print()
    if len(error) != 0:
        print(f"{path} is unreachable")

def delete_web():
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    try:
        with open("web.txt", 'r') as file:
            links = {line.strip() for line in file}
    except FileNotFoundError:
        print("web.txt not found")
    # the below will load each pdf and split into mini documents for embedding
    if '' in links:
        links.remove('')
    deleted = set()
    not_deleted = set()
    error = set()
    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    for path in links:
        value = ''
        try:
            exist = 0
            loader = WebBaseLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique
            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db. skipping its deletion")
                not_deleted.add(path)
                print()
            elif len(existing) == len(hashes):
                print(f"{path} is all inside the database. deleting it in its entirety")
                vectorstore.delete(hashes)
                deleted.add(path)
            else:
                print()
                print(f"{path} partially exists in the db. deleteing its parts that exist")
                print()
                for id in existing:
                    if id not in hashes:
                        position = hashes.index(id)
                        hashes.remove(id)
                        texts.remove(texts[position])
                if len(hashes) == len(existing) and len(hashes) == len(texts):
                    vectorstore.delete(hashes)
                    deleted.add(path)
                else:
                    print(f"error in deleting {path}")
                    not_deleted.add(path)
                    print()
        else:
            print(f"{path} is unreachable")
            error.add(path)
            print()
    if len(deleted) == len(links):
        print("all web pages have been deleted")
        print()
    else:
        print(f"{not_deleted} were not deleted from the database. check the runtime.")
        print()
    if len(error) != 0:
        print(f"{error} is unreachable")

def get_web():
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    try:
        with open("web.txt", 'r') as file:
            links = {line.strip() for line in file}
    except FileNotFoundError:
        print("web.txt not found")

    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    if '' in links:
        links.remove('')
    exists = set()
    non_existing = set()
    error = set()
    meta = set()
    for path in links:
        value = ''
        try:
            exist = 0
            loader = WebBaseLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique
            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db")
                non_existing.add(path)
                print()
            elif len(existing) == len(hashes):
                print()
                print(f"{path} exists in the database")
                exists.add(path)
                print(vectorstore.get(ids = hashes, include = ["metadatas", "documents"]))
                print()
                for t in test['metadatas']:
                    meta.add(t.get("topic"))
            else:
                print(f"{path} only partially exists in database. skipping")
                non_existing.add(path)
        else:
            print(f"error in fetching {path}")
            error.add(path)


    if len(exists) == len(links):
        print("all mentioned webpages are in the database")
        print()
    elif len(exists) == 0 and len(error) == len(links):
        print("could not fetch any url")
        print()
    elif len(exists) == 0:
        print("there are no existing web pages in the database")
        print()
    elif len(non_existing) != 0:
        print(f"{non_existing} are not in the database")
        print()
    
    if len(error) != 0:
        print(f"{error} are unreachable")
        print()
    if len(meta) != 0:
        print(f"metadatas of found documents are {meta}")


def get_pdf():
    # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    loc = input(str("are you working with remote or local pdfs? type remote or local.\n"))
    while True:
        if loc == "remote" or loc == "local":
            break
        else:
            print()
            print("please type 'remote' or 'local' . no capitals")
            print()
    if loc == "local":
        pdf_files = glob.glob("*.pdf")
    else:    
        try:
            with open("pdf.txt", 'r') as file:
                pdf_files = {line.strip() for line in file}
        except FileNotFoundError:
            print("pdf.txt not found")

    vectorstore = Chroma(persist_directory="database")
    # the below will load each pdf and split into mini documents for embedding
    if '' in pdf_files:
        pdf_files.remove('')
    exists = set()
    non_existing = set()
    error = set()
    meta = set()
    for path in pdf_files:
        value = ''
        try:
            exist = 0
            loader = PyPDFLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique
            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db")
                non_existing.add(path)
                print()
            elif len(existing) == len(hashes):
                print()
                print(f"{path} exists in the database")
                exists.add(path)
                print(vectorstore.get(ids = hashes, include = ["metadatas", "documents"]))
                print()
                for t in test['metadatas']:
                    meta.add(t.get("topic"))
            else:
                print(f"{path} only partially exists in database. skipping")
                non_existing.add(path)
        else:
            print(f"error in fetching {path}")
            error.add(path)


    if len(exists) == len(pdf_files):
        print("all mentioned documents are in the database")
        print()
    elif len(exists) == 0 and len(error) == len(pdf_files):
        print("could not fetch any document")
        print()
    elif len(exists) == 0:
        print("none of the documents exist in the database")
        print()
    elif len(non_existing) != 0:
        print(f"{non_existing} are not in the database")
        print()
    
    if len(error) != 0:
        print(f"{error} are unreachable")
        print()
    if len(meta) != 0:
        print(f"metadatas of found documents are {meta}")


def update_web(metadata):
     # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    try:
        with open("web.txt", 'r') as file:
            links = {line.strip() for line in file}
    except FileNotFoundError:
        print("web.txt not found")

    vectorstore = Chroma(persist_directory="database", embedding_function=embeddings)
    
    # the below will load each pdf and split into mini documents for embedding
    if '' in links:
        links.remove('')
    exists = set()
    non_existing = set()
    error = set()
    for path in links:
        value = ''
        try:
            exist = 0
            loader = WebBaseLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique
            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            # updating metadata
            for t in texts:
                m = t.metadata
                m['topic'] = str(metadata)
            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db")
                non_existing.add(path)
                print()
            elif len(existing) == len(hashes):
                print()
                print(f"{path} exists in the database")
                exists.add(path)
                vectorstore.update_documents(hashes, texts)
            else:
                print(f"{path} only partially exists in database. skipping")
                non_existing.add(path)
        else:
            print(f"error in fetching {path}")
            error.add(path)

    if len(exists) == len(links):
        print("all mentioned webpages are updated")
        print()
    elif len(exists) == 0 and len(error) == len(links):
        print("could not fetch any url")
        print()
    elif len(exists) == 0:
        print("there are no existing web pages in the database")
        print()
    elif len(non_existing) != 0:
        print(f"{non_existing} are not in the database")
        print()
    
    if len(error) != 0:
        print(f"{error} are unreachable")

        
def update_pdf(metadata):
     # the below will go through the PDF directory to collect the file paths to the pdfs for usage
    loc = input(str("are you working with remote or local pdfs? type remote or local.\n"))
    while True:
        if loc == "remote" or loc == "local":
            break
        else:
            print()
            print("please type 'remote' or 'local' . no capitals")
            print()
    if loc == "local":
        pdf_files = glob.glob("*.pdf")
    else:    
        try:
            with open("pdf.txt", 'r') as file:
                pdf_files = {line.strip() for line in file}
        except FileNotFoundError:
            print("pdf.txt not found")

    vectorstore = Chroma(persist_directory="database", embedding_function=embeddings)
    # the below will load each pdf and split into mini documents for embedding
    if '' in pdf_files:
        pdf_files.remove('')
    exists = set()
    non_existing = set()
    error = set()
    for path in pdf_files:
        value = ''
        try:
            exist = 0
            loader = PyPDFLoader(path)
            data = loader.load()
        except Exception as e:
            value = '404'

        if value != '404':

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)
            #removing duplicate document content
            unique = []
            for item in texts:
                if item not in unique:
                    unique.append(item)
            texts = unique

            #getting the hashes from the document

            hashes = []

            for t in texts:
                content = t.page_content
                sha3 = hashlib.sha3_256()
                sha3.update(content.encode('utf-8'))
                sha3_hash = sha3.hexdigest()
                hashes.append(sha3_hash)

            # updating metadata
            for t in texts:
                m = t.metadata
                m['topic'] = str(metadata)
            
            test = vectorstore.get(ids = hashes)
            existing = test['ids']
            if len(existing) == 0:
                print()
                print(f"{path} does not exist in db")
                non_existing.add(path)
                print()
            elif len(existing) == len(hashes):
                print()
                print(f"{path} exists in the database")
                exists.add(path)
                vectorstore.update_documents(hashes, texts)
            else:
                print(f"{path} only partially exists in database. skipping")
                non_existing.add(path)
        else:
            print(f"error in fetching {path}")
            error.add(path)


    if len(exists) == len(pdf_files):
        print("all mentioned pdfs are updated")
        print()
    elif len(exists) == 0 and len(error) == len(pdf_files):
        print("could not fetch any pdf")
        print()
    elif len(exists) == 0:
        print("none of the pdfs exist in the database")
        print()
    elif len(non_existing) != 0:
        print(f"{non_existing} are not in the database")
        print()
    if len(error) != 0:
        print(f"{error} are unreachable")
    


if __name__ == "__main__":
    # change the directory to the one your python file is currently in.
    directory = 'D:\Projects\Capstone\Chat-CSEC'
    os.chdir(directory)

    result = load_dotenv('keys.env', verbose=True)

    if result:
        print("environment file specified")
    else:
        print("you do not have an env file with the api key. this wont work")

    OPENAI_API_KEY = os.getenv('key')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    meta_list = ['', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA', 'MITRE']
    include_meta = True
    if result:
        while True:
            datatype = str(input("select the type of data you are working with (type 'pdf' or 'web'):\n"))
            if datatype == "pdf" or datatype == "web":
                break
            else:
                print("please type pdf or web. no capitals either.")
                print()
        while True:
            action = str(input("type upload, delete, update, or get:\n"))
            if action == "upload" or action =='update':
                break
            elif action == "delete" or action == "get":
                include_meta = False
                break
            else:
                print("type upload, get, or delete. not capitals either")
                print()
        if include_meta:
            while True:
                print(meta_list)
                metadata = str(input("what category of malware is this? select one from the list above. if none, then simply hit enter:\n"))
                if metadata in meta_list:
                    break
                else:
                    print("please enter a valid metadata")
                    print()
        if include_meta:
            print(f"to sum this up, your options were {datatype}, {metadata}, and {action}")
            print()
            while True:
                conf = str(input("if you are okay, type yes or no\n"))
                if conf == "yes" or conf == "no":
                    break
                else:
                    print("please type yes or no. no capitals")
        else:
            print(f"to sum this up, your options were {datatype}, and {action}")
            print()
            while True:
                conf = str(input("if you are okay, type yes or no\n"))
                if conf == "yes" or conf == "no":
                    break
                else:
                    print("please type yes or no. no capitals")

            
        if conf == "yes" and datatype == "pdf" and action == "upload":
            upload_pdf(metadata)
        elif conf == "yes" and datatype == "pdf" and action == "update":
            update_pdf(metadata)
        elif conf == "yes" and datatype == "pdf" and action == "delete":
            delete_pdf()
        elif conf == "yes" and datatype == "pdf" and action == "get":
            get_pdf()
        elif conf == "yes" and datatype == "web" and action == "upload":
            upload_web(metadata)
        elif conf == "yes" and datatype == "web" and action == "update":
            update_web(metadata)
        elif conf == "yes" and datatype == "web" and action == "delete":
            delete_web()
        elif conf == "yes" and datatype == "web" and action == "get":
            get_web()
        else:
            print("run the program again and be sure of your answers next time.")
            