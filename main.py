from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
import getpass
import os

if os.getenv("MISTRAL_API_KEY") is None:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass(
        "Please set the MISTRAL_API_KEY environment variable: "
    )
    print("MISTRAL_API_KEY set successfully.")

embeddings = MistralAIEmbeddings(model="mistral-embed")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


def load_text_file(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)


print(vector_store.similarity_search("async"))

# doc1 = Document(
#     page_content="This is a test document.",
#     metadata={},
# )

# vector_store.add_documents([doc1], ids=["doc1"])
vector_store.delete(ids=["doc1"])
print(vector_store.similarity_search("async"))
