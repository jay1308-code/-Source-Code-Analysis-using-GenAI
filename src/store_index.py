from src.helper import repo_ingestion,load_repo,create_text_chunks,load_embeddings
from dotenv import load_dotenv
from src.constant import *
from langchain.vectorstores import FAISS
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

documents = load_repo(REPO_PATH)
text_chunks = create_text_chunks(documents)
embeddings = load_embeddings()

#Storing the vectors in FISSA
vector_store= FAISS.from_documents(text_chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

