## Tab 1
# Building LLM applications LCEL is used to combine chains

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key

## Tab 2
!pip install langchain_groq

## Tab 3
from langchain_groq import ChatGroq #Groq works without install any model just by using api key
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
model

## Tab 4
!pip install langchain_core

## Tab 5
from langchain_core.messages import HumanMessage,SystemMessage #Instructs model how to work

messages = [
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content="Hello How are you?")
]
result = model.invoke(messages)

## Tab 6
result

## Tab 7
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
parser.invoke(result)

## Tab 8
#Using LCEL- chain the components
chain=model|parser
chain.invoke(messages)

## Tab 9
##Prompt Templates
from langchain_core.prompts import ChatPromptTemplate
generic_template="Translate the following into {language}:"

prompt =ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

## Tab 10
result=prompt.invoke({"language":"French","text":"Hello"})

## Tab 11
result.to_messages()

## Tab 12
#Chaining together components with LCEL
chain=prompt|model|parser
chain.invoke({"language":"French","text":"Hello"})