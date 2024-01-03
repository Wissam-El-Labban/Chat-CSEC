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
from streamlit.delta_generator import DeltaGenerator
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document
from typing import List
# change the directory to the one your python file is currently in.
directory = 'D:\Projects\Capstone\Chat-CSEC'
os.chdir(directory)

result = load_dotenv('keys.env', verbose=True)

if result:
    print("environment file specified")
else:
    print("you do not have an env file with the api key. this wont work")

if result:
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

    # directory full path

    OPENAI_API_KEY = os.getenv('key')


    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    model_name = 'gpt-3.5-turbo'
    #model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    msg = StreamlitChatMessageHistory()

    memory=ConversationBufferMemory(memory_key="chat_history",chat_memory= msg, input_key='question', output_key= 'answer', return_messages=True)


    vectordb = Chroma(persist_directory='database', embedding_function=embeddings)
    model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True)

    # regular prompt template

    prompt_template = """You are a malware expert that answers in very high detail. Get every technical detail and explain as if you are talking to an expert. If you dont have the answer, say 'I dont have this information'.  
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


    #depth = st.selectbox('this is how in depth the knowlegebase used should be. select high for a higher level overview, or deep for in depth.', ('high', 'deep'))
    with st.sidebar:
        #topic = st.selectbox('What topic of malware would you like to specificalyl search for? you can choose general for all purpose (recommended). You can even chat with MITRE Content.', ('general', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA', 'MITRE'))
        topic = st.selectbox('What topic of malware would you like to specificalyl search for? you can choose general for all purpose (recommended). You can even chat with MITRE Content.', ('general', 'ICS SCADA', 'MITRE'))
        guess = st.selectbox('Cant answer a question? would you like the llm contrsuct one? It is your responsibility to verify that the data is accurate', ('no', 'yes'))
        clear = st.button('token issues? Push to clear memory')
        if clear == True:
            memory.clear()

    if str(topic) == 'general' and str(guess) == 'no':
        Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
        Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, chain_type="stuff", max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    elif str(topic) == 'general' and str(guess) == 'yes':
        Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
        Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, chain_type="stuff", max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    elif str(guess) == 'yes':
        Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
        Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT2}, chain_type="stuff", max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
    elif str(guess) == 'no':
        filter = {'topic': str(topic) }
        Malware_context = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={"filter": filter, 'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)
        Malware_exception = ConversationalRetrievalChain.from_llm(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={"filter": filter, 'k': 15}), memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT}, chain_type="stuff",max_tokens_limit = 4090, return_source_documents = True, return_generated_question = True)

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
                response = Malware_context({"question": prompt}, return_only_outputs=True, callbacks=[stream_handler])
                st.session_state.messages.append(ChatMessage(role="assistant", content=response['answer']))
                print()
                documents = response["source_documents"]
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
                print(f"rephrased question: {response['generated_question']}")
                print()
                print("answer:")
                print(response["answer"])
                print()

            except Exception as E:
                try:
                    stream_handler = StreamHandler(st.empty())
                    response = Malware_exception({"question": prompt}, return_only_outputs=True, callbacks=[stream_handler])
                    st.session_state.messages.append(ChatMessage(role="assistant", content=response['answer']))
                    print()
                    documents = response["source_documents"]
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
                    print(f"rephrased question: {response['generated_question']}")
                    print()
                    print("answer:")
                    print(response["answer"])
                    print()
                except Exception as E:
                    st.write(str(E)+"\n clearing memory")
                    # clearing memory in case of token overload error
                    memory.clear()
