"""
RAG Engine - Core retrieval-augmented generation engine
Uses ChromaDB for vector storage and sentence-transformers for embeddings
Uses Gemini for LLM synthesis (offline embeddings, online LLM)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import (
    Settings as LlamaSettings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger("RAGEngine")


class RAGEngine:
    """
    Production-ready RAG engine with ChromaDB and local embeddings
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "iabisa_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize RAG engine with ChromaDB and sentence-transformers

        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: ChromaDB collection name
            embedding_model: HuggingFace embedding model name
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(f"Initializing RAG Engine with model: {embedding_model}")

        # Initialize embedding model
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                cache_folder=str(self.persist_dir / "models"),
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Configure LlamaIndex settings with Gemini LLM
        LlamaSettings.embed_model = self.embed_model

        # Use Gemini LLM if API key available (for query synthesis)
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            try:
                LlamaSettings.llm = Gemini(
                    model="models/gemini-flash-lite-latest",
                    api_key=gemini_api_key,
                )
                logger.info("Gemini LLM configured for RAG synthesis")
            except Exception as e:
                logger.warning(
                    f"Failed to configure Gemini LLM: {e}, using retrieval-only mode"
                )
                LlamaSettings.llm = None
        else:
            logger.warning("No Gemini API key found, using retrieval-only mode")
            LlamaSettings.llm = None

        LlamaSettings.chunk_size = chunk_size
        LlamaSettings.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(f"ChromaDB initialized at: {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Using collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise

        # Initialize vector store and index
        self._initialize_index()

        # Text splitter for chunking
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info("RAG Engine initialized successfully")

    def _initialize_index(self):
        """Initialize or load the vector store index"""
        try:
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Try to load existing index or create new one
            if self.collection.count() > 0:
                self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                logger.info(
                    f"Loaded existing index with {self.collection.count()} documents"
                )
            else:
                self.index = VectorStoreIndex(
                    [],
                    storage_context=self.storage_context,
                )
                logger.info("Created new empty index")

        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise

    def index_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Index documents from file paths

        Args:
            file_paths: List of file paths to index
            metadata: Optional metadata to attach to documents

        Returns:
            Dictionary with indexing results
        """
        try:
            documents = []
            for file_path in file_paths:
                path = Path(file_path)
                if not path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue

                # Load document
                try:
                    loader = SimpleDirectoryReader(
                        input_files=[str(path)],
                        file_metadata=lambda _: metadata or {},
                    )
                    docs = loader.load_data()
                    documents.extend(docs)
                    logger.info(f"Loaded document: {path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

            if not documents:
                return {
                    "success": False,
                    "message": "No documents loaded",
                    "indexed_count": 0,
                }

            # Split into nodes and index
            nodes = self.text_splitter.get_nodes_from_documents(documents)

            # Add to index
            for node in nodes:
                self.index.insert_nodes([node])

            count = len(nodes)
            logger.info(
                f"Successfully indexed {count} chunks from {len(documents)} documents"
            )
            count = len(nodes)
            logger.info(
                f"Successfully indexed {count} chunks from {len(documents)} documents"
            )

            return {
                "success": True,
                "message": f"Indexed {len(documents)} documents into {count} chunks",
                "indexed_count": len(documents),
                "chunk_count": count,
            }

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return {
                "success": False,
                "message": f"Indexing failed: {str(e)}",
                "indexed_count": 0,
            }

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Query the RAG engine for relevant context

        Args:
            query_text: Query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            Dictionary with query results and sources
        """
        try:
            if self.collection.count() == 0:
                return {
                    "success": False,
                    "message": "No documents indexed yet",
                    "context": "",
                    "sources": [],
                }

            # Use retriever directly (no LLM synthesis needed)
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
            )

            # Retrieve nodes
            nodes = retriever.retrieve(query_text)

            # Extract source nodes
            sources = []
            context_parts = []

            for idx, node in enumerate(nodes, 1):
                score = node.score if hasattr(node, "score") else 1.0

                # Filter by similarity threshold
                if score < similarity_threshold:
                    continue

                text = node.node.get_content()
                metadata = node.node.metadata

                source_info = {
                    "rank": idx,
                    "text": text,
                    "score": float(score),
                    "metadata": metadata,
                }
                sources.append(source_info)
                context_parts.append(f"[Source {idx}]: {text}")

            # Combine context
            context = "\n\n".join(context_parts) if context_parts else ""

            return {
                "success": True,
                "context": context,
                "sources": sources,
                "query": query_text,
                "retrieved_count": len(sources),
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "success": False,
                "message": f"Query failed: {str(e)}",
                "context": "",
                "sources": [],
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG engine"""
        try:
            return {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "persist_dir": str(self.persist_dir),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialize_index()
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(
    persist_dir: str = "./data/chroma",
    collection_name: str = "iabisa_docs",
    reset: bool = False,
) -> RAGEngine:
    """
    Get or create singleton RAG engine instance

    Args:
        persist_dir: Directory to persist ChromaDB data
        collection_name: ChromaDB collection name
        reset: Force reinitialize the engine

    Returns:
        RAGEngine instance
    """
    global _rag_engine

    if _rag_engine is None or reset:
        logger.info("Initializing new RAG engine instance")
        _rag_engine = RAGEngine(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    return _rag_engine
