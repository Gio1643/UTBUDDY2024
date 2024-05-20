from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# create a vector db
def create_vector_database():
  loader = DirectoryLoader(DATA_PATH, glob = '*.pdf',loader_cls = PyPDFLoader)
  documents = loader.load()
  text_splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 10)
  texts = text_splitters.split_documents(documents)
  embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-V2', model_kwargs = {'device':'cpu'})
  db = FAISS.from_documents(texts, embeddings)
  db.save_local(DB_FAISS_PATH)
  
if __name__ == '__main__':
  create_vector_database()