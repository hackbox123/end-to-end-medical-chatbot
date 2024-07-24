from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)


extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embedding=download_hugging_face_embeddings()


os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")
index_name="medical-chatbot"

#creating embeddings for each of the text chunks and storing
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name)