## Tab 1
# Building a Chatbot The chatbot will be able to have conversations and remember previous interactions

import os
from dotenv import load_dotenv
load_dotenv() #Load all the environment variables

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key

## Tab 2
from langchain_groq import ChatGroq #Load model from groq
model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
model

## Tab 3
from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi,I am Amritanshu ,studying in Banglore")])

## Tab 4
from langchain_core.messages import AIMessage
model.invoke(
    [
       HumanMessage(content="Hi,I am Amritanshu ,studying in Banglore"),
       AIMessage(content="Hi Amritanshu,\n\nIt's great to meet you! Bangalore is a fantastic city. What are you studying there?"),
       HumanMessage(content="Hey what's my name and where do i study?")
    ]
)

## Tab 5
## Message History
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store={}

#for different users to have different message history,isme string pass hoga
def get_session_history(session_id:str)->BaseChatMessageHistory: ##Return type is chathistory
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(model,get_session_history)#to interact with our llm model based on chat history 

## Tab 6
config={"configurable":{"session_id":"chat1"}}

## Tab 7
response=with_message_history.invoke(
    [HumanMessage(content="Hi,I am Amritanshu ,studying in Banglore")],
    config=config ##matlab session id chat1 ke liye it will remember the context
)

## Tab 8
response.content

## Tab 9
with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config ##matlab session id chat1 ke liye it will remember the context
)

## Tab 10
##Change the config(changing the session id)
config1={"configurable":{"session_id":"chat2"}}
response=with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config1
)
response.content

## Tab 11
response=with_message_history.invoke(
    [HumanMessage(content="My name is John")],
    config=config1
)
response.content

## Tab 12
response=with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config1
)
response.content  #with this config id we are able to switch with the contexts

## Tab 13
## Prompt Templates

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Answer all the questions to the best of your ability"),
        MessagesPlaceholder(variable_name="messages")##inside this give infomfrom human side in messages
    ]
)
chain = prompt|model

## Tab 14
chain.invoke({"messages":[HumanMessage(content="Hi,my name is Amritanshu")]})

## Tab 15
with_message_history=RunnableWithMessageHistory(chain,get_session_history)

## Tab 16
config = {"configurable":{"session_id":"chat3"}}
response=with_message_history.invoke(
    [HumanMessage(content="Hi,my name is Amritanshu")],
    config=config,
)
response.content

## Tab 17
#Add more complexity
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Answer all the questions to the best of your ability in {language}"),
        MessagesPlaceholder(variable_name="messages")##inside this give infomfrom human side in messages
    ]
)
chain = prompt|model

## Tab 18
response = chain.invoke({"messages":[HumanMessage(content="Hi My name is Amritanshu")],"language":"Hindi"})
response.content

## Tab 19
with_message_history=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

## Tab 20
config={"configurable":{"session_id":"chat4"}}
response = with_message_history.invoke(
    {'messages':[HumanMessage(content="Hi,I am Amritanshu")],"language":"Hindi"},
    config=config
)
response.content

## Tab 21
response = with_message_history.invoke(
    {'messages':[HumanMessage(content="What's my name?")],"language":"Hindi"},
    config=config
)
response.content

## Tab 22
# Manage the Conversation History trim message:helper to reduce how many messages u are sending to the model

from langchain_core.messages import SystemMessage,trim_messages

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",  # Corrected typo here
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you are a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2+2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!")
]

trimmer.invoke(messages)

## Tab 23
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
    |prompt
    |model
)
response=chain.invoke(
    {"messages":messages+[HumanMessage(content="What ice cream do i like")],
    "language":"English"}
)
response.content

## Tab 24
response=chain.invoke(
    {"messages":messages+[HumanMessage(content="What math problem did i ask")],
    "language":"English"}
)
response.content

## Tab 25
#Lets wrap this in the message history
with_message_history=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable":{"session_id":"chat5"}}

## Tab 26
response=with_message_history.invoke(
    {"messages":messages+[HumanMessage(content="What's my name")],
    "language":"English"},
    config=config
)
response.content

## Tab 27
# Vector stores and retrievers

# Documents

from langchain_core.documents import Document

documents=[
    Document(
        page_content="Dogs are great companions,known for their loyalty and friendliness",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="Cats and independent pets that often enjoy their own space",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pet for beginners,requiring relatively simple care.",
        meta_data={"source":"fish-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        meta_data={"source":"mammal-pets-doc"},
    ),
]

## Tab 28
documents

## Tab 29
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
llm

## Tab 30
#Vector db me store karne se pehle embeddding karni padhti hai
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Tab 31
#Vector stores
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embedding=embeddings)

## Tab 32
vectorstore.similarity_search("cat")

## Tab 33
##Async query
await vectorstore.asimilarity_search("cat")

## Tab 34
vectorstore.similarity_search_with_score("cat")

## Tab 35
# Retrievers

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda #retrievers are runnables

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
retriever.batch(["cat","dog"]) #vector store db ko retrievers me convert kar diya

## Tab 36
retriever = vectorstore.as_retriever(  # document->vectorstore->retriever
    search_type="similarity",
    search_kwargs={"k":1}
)
retriever.batch(["cat","dog"])  #other rway to convert to retriver(best way)

## Tab 37
#this is rag (retrieval augumented generation)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message="""
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human",message)])

rag_chain = {"context":retriever,"question":RunnablePassthrough()}|prompt|llm
response=rag_chain.invoke("tell me about dogs")
print(response.content)