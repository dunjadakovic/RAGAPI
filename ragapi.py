from flask import Flask, request, jsonify, make_response
from flask_caching import Cache 
import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import requests
import logging

app = Flask(__name__)

# Set API key for OpenAI usage

app.config["SECRET_KEY"] = os.urandom(24)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load CSV file
file_url = "https://raw.githubusercontent.com/dunjadakovic/RAGAPI/main/ContentAndCategories.csv"
response = requests.get(file_url)
with open("temp.csv", "wb") as f:
    f.write(response.content)

loader = CSVLoader(file_path="temp.csv")
data = loader.load()
# Create text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,  # length of longest category
    chunk_overlap=0,  # no overlap as complete separation, structured data
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(data)

# Create vector store
vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Define prompt template
template = """Use the following pieces of context to answer the question at the end.
Use as many of the provided words as possible to make a sentence. Make sure the sentence is child-safe and appropriate.
Don't say anything that isn't a direct part of your answer. Take out one word from the sentence. The word must be in the provided list. 
Replace it with ______. Then, separate the next part from the sentence
with a newline (\n). Take the word you replaced with ______ and add two other words separated by comma. The two other
words have to be in a similar semantic/syntactic category as the replaced word but must show some small differences.
Provide the sentence, then a newline (\n) and then the three words as described. Do not provide anything else. 
{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | custom_rag_prompt
      | llm
      | StrOutputParser()
    )
@app.route('/api/get_sentence', methods=['GET'])
def getSentence():
    result = request.cookies.get('result')
    if result:
      resultList = result.split("\n")
      resultSentence = resultList[0]
      return jsonify(resultSentence)
    else:
      return jsonify({'error': 'No result found, please check internet connection'}), 404
@app.route('/api/option1', methods=['GET'])
def getOption1():
  result = request.cookies.get('result')
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption1 = resultOptions[0]
        return jsonify(resultOption1)
  else:
      return jsonify({"error": "No result found, please check internet connection"}), 404
@app.route('/api/option2', methods=['GET'])
def getOption2():
  result = request.cookies.get('result')
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption2 = resultOptions[1]
        return jsonify(resultOption2)
  else:
        return jsonify({"error": "No result found, please check internet connection"}), 404
@app.route('/api/option3', methods=['GET'])
def getOption3():
  result = request.cookies.get('result')
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption3 = resultOptions[2]
        return jsonify(resultOption3)
  else:
        return jsonify({"error": "No result found, please check internet connection"}), 404
@app.route('/api/regenerate', methods=['GET'])
def generate_sentence():
    level = request.args.get('level')
    topic = request.args.get('topic')
    if not level or not topic:
        return jsonify({'error': 'Missing level or topic'}), 400  
    else:
        stringConcat = level + "," + topic
        resultChain = rag_chain.invoke(stringConcat)
        while "_" not in resultChain:
            resultChain = rag_chain.invoke(stringConcat)
        logging.info(f"Level: {level} Topic {topic}")
        resp = jsonify({'sentence': resultChain})
        resp.set_cookie('result', resultChain, max_age=60)
        return resp
if __name__ == '__main__':
    app.run(debug=True)

