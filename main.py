from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

load_dotenv()

def initialize_custom_conversational_chain(file_path):
    # Set your OpenAI API key
    os.environ.get('OPENAI_API_KEY')

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(documents)

    db = Chroma.from_documents(documents,
                               HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    retriever = db.as_retriever(search_kwargs={'k': 3})

    system_template = """I want you to act as a chat with pdf customer support representative. Answer the user's 
            question using the given context in a concise response of no more than 50 tokens. If the answer cannot be found 
            in the context, simply state, " I don't know the answer." also if the query is as greetings (i.e.Hi, Hello,How are you?)simply state, " I don't know the answer.". 
            ---------------- {context}"""

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Create conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.2),
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return qa_chain


def custom_answer_query(query, path):
    chat_history = []
    qa_chain = initialize_custom_conversational_chain(path)
    history_query = chat_history[-2:] if len(chat_history) >= 2 else chat_history[:]
    result = qa_chain({'question': query, 'chat_history': history_query})
    chat_history.append((query, result['answer']))

    answer = result['answer']
    metadata_list = []
    if answer == "I don't know the answer.":
        metadata_list.append('{Not found}')
    else:
        for document in result['source_documents']:
            metadata_list.append(document.metadata)

    return answer, metadata_list







if __name__ == "__main__":
    while True:
        user_query = input("Enter query! ")
        response = custom_answer_query(user_query, "data/data_HR/files/Code_of_Conduct.pdf")
        print(response)
