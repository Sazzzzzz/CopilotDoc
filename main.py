from typing import List
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from bs4 import Tag
from langchain_community.document_loaders import DirectoryLoader
import itertools
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from pprint import pprint

# cSpell:words qdrant

EMBEDDINGS = MistralAIEmbeddings(model="mistral-embed")
VECTOR_STORE_PATH = "langchain_qdrant"
CLIENT = QdrantClient(path=VECTOR_STORE_PATH)


class Collection:
    def __init__(self, name: str, embeddings: Embeddings = EMBEDDINGS) -> None:
        self.name = name
        self.embeddings = embeddings


class BuildEngine:
    """
    A class to build a semantic search engine using MistralAI embeddings.
    """

    def __init__(self, embeddings: Embeddings) -> None:
        """
        Initialize the environment variables for MistralAI API key.
        """
        self.embeddings = embeddings

    def load_docs(self, path: str) -> list[Document]:
        return DirectoryLoader(
            path=path, glob="**/*.html", show_progress=True, silent_errors=True
        ).load()

    def split_markdown(
        self, docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100
    ) -> List[Document]:
        text_splitter = MarkdownHeaderTextSplitter(
            [
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ],
            strip_headers=False,
        )

        all_splits = list(
            itertools.chain.from_iterable(
                text_splitter.split_text(doc.page_content) for doc in docs
            )
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

        return text_splitter.split_documents(all_splits)

    def split_html(self, docs: list[Document]) -> List[Document]:
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ]

        def code_handler(element: Tag) -> str:
            data_lang = element.get("data-lang")
            code_format = f"<code:{data_lang}>{element.get_text()}</code>"
            return code_format

        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            max_chunk_size=50,
            preserve_images=True,
            preserve_videos=True,
            elements_to_preserve=["table", "ul", "ol", "code"],
            denylist_tags=["script", "style", "head"],
            custom_handlers={"code": code_handler},
        )
        all_splits = list(
            itertools.chain.from_iterable(
                splitter.split_text(doc.page_content) for doc in docs
            )
        )
        return all_splits

    def build_collection(
        self, collection_name: str, docs: list[Document]
    ) -> Collection:
        """
        Build a collection in Qdrant using the provided documents.
        """
        CLIENT.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        self.vector_store = QdrantVectorStore(
            client=CLIENT,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        self.vector_store.add_documents(documents=docs)
        return Collection(name=collection_name, embeddings=self.embeddings)


class SearchEngine:
    """
    A semantic search engine that uses MistralAI embeddings for document retrieval.
    Currently support markdown files
    """

    def __init__(self, collection: Collection) -> None:
        """
        Initialize the environment variables for MistralAI API key.
        """
        self.vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=collection.name,
            embedding=collection.embeddings,
            path=VECTOR_STORE_PATH,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def query(self, query: str) -> list[Document]:
        """
        Query the vector store for similar documents.
        """
        results = self.retriever.invoke(query)
        return results


b = BuildEngine(embeddings=EMBEDDINGS)
docs = b.load_docs("manim_doc")
b.split_html(docs)
collection = b.build_collection("manim_doc", docs)
s = SearchEngine(collection=collection)
results = s.query("How to move objects?")
