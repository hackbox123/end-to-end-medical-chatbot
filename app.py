from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embedding=download_hugging_face_embeddings()

# pc = Pinecone(PINECONE_API_KEY)
# index = pc.Index("medical-chatbot")
index_name="medical-chatbot"

#loading the index

docsearch=PineconeVectorStore.from_existing_index(index_name, embedding)



PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model\llama-2-7b.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':2000,
                          'temperature':0.75,
                          'context_length': 8000})


retriever1 = docsearch.as_retriever(search_kwargs={'k': 3})
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever1,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])




if __name__ =='__main__':
    app.run(debug=True)