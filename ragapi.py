from flask import Flask, request, jsonify, session
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

app = Flask(__name__)

# Set API key for OpenAI usage

app.config["SECRET_KEY"] = os.urandom(24)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
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
Use as many of the provided words as possible to make a sentence. Make sure the sentences are child-safe and appropriate.
Don't say anything that isn't a direct part of your answer. Write three sentences. Furthermore, please take out one word
from the sentence that is also in the provided list. Replace it with ______. Then, separate the next part from the sentence
with a newline (\n) and then provide the word you replaced and two additional words of the same syntactic category that are
also in the provided list. If the word is a verb, make sure all the verbs are conjugated appropriately. Alternatively, if the chosen
word is a verb, you can also put the same word in three types of conjugations or spellings. Make sure the three words are comma separated.
Do not write any more sentences than that
{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
@app.before_request
def generate_sentence():
    session.permanent = True
    if 'reset' in request.args and request.args['reset'] == 'true':
        session.pop('result', None)  # Reset the session
    if 'result' not in session:
        try:
            level = request.args.get('level')
            topic = request.args.get('topic')
            if not level or not topic:
                return jsonify({'error': 'Missing level or topic'}), 400
            result = rag_chain.invoke(level, topic)
            return jsonify({'sentence': result})
        except Exception as e:
            logging.error(f'Error generating sentence: {e}')
            return jsonify({'error': 'Internal Server Error'}), 500
        rag_chain = (
          {"context": retriever | format_docs, "question": RunnablePassthrough()}
          | custom_rag_prompt
          | llm
          | StrOutputParser()
        )
        result = rag_chain.invoke(level, topic)
        session["result"] = result
@app.route('/api/get_sentence', methods=['GET'])
def getSentence():
    result = session.get("result")
    if result:
      resultList = result.split("\n")
      resultSentence = resultList[0]
      return jsonify({resultSentence})
    else:
      return jsonify({'error': 'No result found, please check internet connection'}), 404
@app.route('/api/option1', methods=['GET'])
def getOption1():
  result = session.get("result")
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption1 = resultOptions[0]
        return jsonify({resultOption1})
  else:
      return jsonify({"error": "No result found, please check internet connection"}), 404
@app.route('/api/option2', methods=['GET'])
def getOption2():
  result = session.get("result")
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption2 = resultOptions[1]
        return jsonify({resultOption2})
  else:
        return jsonify({"error": "No result found, please check internet connection"}), 404
@app.route('/api/option3', methods=['GET'])
def getOption3():
  result = session.get("result")
  if result:
        resultList = result.split("\n")
        resultOptions = resultList[1].split(",")
        resultOption3 = resultOptions[2]
        return jsonify({resultOption3})
  else:
        return jsonify({"error": "No result found, please check internet connection"}), 404
if __name__ == '__main__':
    app.run(debug=True)

