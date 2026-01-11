## Tab 1
# Document Ingestion - Document Loaders

##Text loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader('speech.txt')
loader

## Tab 2
text_documents = loader.load()
text_documents

## Tab 3
## Reading a pdf file
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('attention1.pdf')
docs = loader.load()
docs

## Tab 4
type(docs[0])

## Tab 5
## Web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title","post-content","post-header") ##website ke andar jo field chahiye
                       ))
                       )

## Tab 6
loader.load()

## Tab 7
##Arxiv
from langchain_community.document_loaders import ArxivLoader ##Similarily u can use any other document loader
docs=ArxivLoader(query="1706.03762",load_max_docs=2).load()
len(docs)

## Tab 8
docs

## Tab 9
# Text Splitting From Documents

## Reading a pdf file
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('attention1.pdf')
docs = loader.load()
docs

## Tab 10
# How to recursively split text by characters(Text splitting from documents-recursive character text splitter)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
##when dividing document sinto smaller chunks of text each should have a chunk size
##of 500, and a overlap of 50 characters
final_docs=text_splitter.split_documents(docs)
final_docs

## Tab 11
print(final_docs[0])
print(final_docs[1])

## Tab 12
##Text Loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader('speech.txt')
docs=loader.load()
docs

## Tab 13
speech=""  ##Reading the document directly
with open("speech.txt") as f:
    speech = f.read()
speech

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)
text = text_splitter.create_documents([speech])
print(text)

## Tab 14
print(text[0])
print(text[1])

## Tab 15
# Character text splitter

from langchain_community.document_loaders import TextLoader
loader = TextLoader('speech.txt')
docs=loader.load()
docs

## Tab 16
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)
text_splitter.split_documents(docs)

## Tab 17
speech=""  ##Reading the document directly
with open("speech.txt") as f:
    speech = f.read()
speech

text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=10)
text = text_splitter.create_documents([speech])
print(text[0])
print(text[1])

## Tab 18
# How to Split by HTML header

from langchain_text_splitters import HTMLHeaderTextSplitter
html_string = """ <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple HTML Page</title>
</head>
<body>

    <header>
        <h1>Welcome to My Simple Webpage</h1>
    </header>

    <main>
        <h2>About This Page</h2>
        <p>This is a basic HTML-only page with a header, content section, and footer.</p>
        <p>Feel free to edit this content as you like.</p>
    </main>

    <footer>
        <p>&copy; 2024 My Simple Webpage</p>
    </footer>

</body>
</html>
"""
headers_to_split_on =[
    ("h1","Header 1"),
    ("h2","Header 2"),
    ("h3","Header 3")
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits=html_splitter.split_text(html_string)
html_header_splits

## Tab 19
url = "https://plato.stanford.edu/"

headers_to_split_on =[
    ("h1","Header 1"),
    ("h2","Header 2"),
    ("h3","Header 3")
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits=html_splitter.split_text_from_url(url)
html_header_splits

## Tab 20
# How to split JSON data

import json
import requests

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

## Tab 21
json_data

## Tab 22
from langchain_text_splitters import RecursiveJsonSplitter
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_json(json_data)

## Tab 23
json_chunks

## Tab 24
for chunk in json_chunks[:3]:
    print(chunk)

## Tab 25
#The splitter can also output documents
docs = json_splitter.create_documents(texts=[json_data])
for doc in docs[:3]:
    print(doc)

## Tab 26
texts = json_splitter.split_text(json_data)
print(texts[0])
print(texts[1])

## Tab 27
# Embedding Techniques -Converting text into vectors

import os
from dotenv import load_dotenv
load_dotenv() ##load all the environment varaibles

## Tab 28
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Tab 29
from langchain_openai import OpenAIEmbeddings
#to convert text into vector we use embeddings
embeddings =OpenAIEmbeddings(model="text-embedding-3-large")
embeddings

## Tab 30
text = "This is a tutorial on OPENAI embedding"
query_result = embeddings.embed_query(text)
query_result  ##this will take cost to run(u have to pay)

## Tab 31
query_result[0] ##this will give us dimensions
len(query_result)

## Tab 32
embeddings_1024 =OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)
embeddings_1024

## Tab 33
text = "This is a tutorial on OPENAI embedding"
query_result = embeddings_1024.embed_query(text)
len(query_result)
query_result

## Tab 34
##Text loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader('speech.txt')
docs =loader.load()
docs

## Tab 35
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
##when dividing document sinto smaller chunks of text each should have a chunk size
##of 500, and a overlap of 50 characters
final_documents=text_splitter.split_documents(docs)
final_documents

## Tab 36
# Chunk of document liya ab isko convert into vector and store it in vector db

##Vector Embedding and Vector StoreDB
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(final_documents,embeddings_1024)
db

## Tab 37
##Retrieve the results from querying vector store db
query = "For the process of speaking to a group of people, see Public speaking. For other uses, see Speech (disambiguation)."
retrieved_results=db.similarity_search(query)#searching the above text in vector db(chroma)
print(retrieved_results)

## Tab 38
# OLLAMA

from langchain_community.embeddings import OllamaEmbeddings

## Tab 39
embeddings=(
    OllamaEmbeddings(model="gemma:2b") #By default it uses llama2
)

## Tab 40
embeddings

## Tab 41
r1=embeddings.embed_documents(
    [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
    ]
)

## Tab 42
r1

## Tab 43
r1[0]

## Tab 44
len(r1[0]) #This will give number of dimensions

## Tab 45
embeddings.embed_query("What is the second letter of greek alphabet")

## Tab 46
##Other embeddings model
### "https://ollama.com/blog/embedding-models"

## Tab 47
embeddings = OllamaEmbeddings(model="mxbai-embed-large")#this model needs to be downloaded
text = "This is a test document."
query_result = embeddings.embed_query(text)
query_result

## Tab 48
len(query_result)

## Tab 49
# Embedding Techniques Using HuggingFace

import os
from dotenv import load_dotenv
load_dotenv() #Load all the environment variables

## Tab 50
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

## Tab 51
# Sentence Transformers on Hugging Face

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Tab 52
text="this is a test document"
query_result=embeddings.embed_query(text)
query_result

## Tab 53
len(query_result)

## Tab 54
doc_result = embeddings.embed_documents([text,"This is not a test document"])
doc_result[0]

## Tab 55
# Vector Store DB

# FAISS

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("speech.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200,chunk_overlap=30)
docs= text_splitter.split_documents(documents)

## Tab 56
docs

## Tab 57
embeddings = OllamaEmbeddings()
db = FAISS.from_documents(docs,embeddings)
db

## Tab 58
##querying
query="Speech production visualized by real-time MRI"
docs = db.similarity_search(query)
docs

## Tab 59
docs[0].page_content

## Tab 60
# As a Retriever We can also convert the vectorstore into a Retriever class. This allows us to easily use it in other LangChain methods,which largely work with retrievers

retriever=db.as_retriever()
docs=retriever.invoke(query)
docs[0].page_content

## Tab 61
# Similarity Search with Score

docs_and_score = db.similarity_search_with_score(query)
docs_and_score

## Tab 62
embedding_vector = embeddings.embed_query(query)
embedding_vector

## Tab 63
docs_score=db.similarity_search_by_vector(embedding_vector)
docs_score

## Tab 64
## Saving and Loading
db.save_local("faiss_index")

## Tab 65
new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
docs=new_db.similarity_search(query)

## Tab 66
docs

## Tab 67
# Chroma

#building a sample vector db
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Tab 68
loader = TextLoader("speech.txt")
data = loader.load()
data

## Tab 69
#Split
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
splits = text_splitter.split_documents(data)

## Tab 70
embedding = OllamaEmbeddings()
vectordb = Chroma.from_documents(documents=splits,embedding=embedding)
vectordb

## Tab 71
##query it
query = "Speech production visualized by real-time MRI"
docs = vectordb.similarity_search(query)
docs[0].page_content

## Tab 72
##Saving to the disk
vectordb = Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db")

## Tab 73
#load from disk
db2 = Chroma(persist_directory="./chroma_db",embedding_function=embedding)
docs=db2.similarity_search(query)
print(docs[0].page_content)

## Tab 74
##similarity search with score
docs = vectordb.similarity_search_with_score(query)
docs

## Tab 75
##Retriever option
retriever=vectordb.as_retriever()
retriever.invoke(query)[0].page_content
