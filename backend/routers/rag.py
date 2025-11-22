"""
RAG Router - FastAPI endpoints for document management and RAG queries
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import get_rag_engine

logger = logging.getLogger("RAGRouter")

router = APIRouter(prefix="/rag", tags=["RAG"])

# Constants
UPLOAD_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# ==================== MODELS ====================


class RAGQueryRequest(BaseModel):
    """Request model for RAG query"""

    query: str = Field(..., min_length=1, description="Query text")
    top_k: int = Field(3, ge=1, le=10, description="Number of results to retrieve")
    similarity_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class RAGQueryResponse(BaseModel):
    """Response model for RAG query"""

    success: bool
    context: str
    sources: List[dict]
    query: str
    retrieved_count: int
    message: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information model"""

    id: str
    filename: str
    size: int
    upload_date: str
    file_type: str


class UploadResponse(BaseModel):
    """Response model for document upload"""

    success: bool
    message: str
    indexed_count: int
    chunk_count: int
    files: List[DocumentInfo]


class StatsResponse(BaseModel):
    """Response model for RAG statistics"""

    collection_name: str
    total_documents: int
    persist_dir: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    rag_enabled: bool
    documents_indexed: int
    message: str


# ==================== ENDPOINTS ====================


@router.get("/health", response_model=HealthResponse)
async def rag_health_check():
    """
    Check RAG system health and status
    """
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()

        return HealthResponse(
            status="healthy",
            rag_enabled=True,
            documents_indexed=stats.get("total_documents", 0),
            message="RAG system is operational",
        )
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            rag_enabled=False,
            documents_indexed=0,
            message=f"RAG system error: {str(e)}",
        )


@router.get("/stats", response_model=StatsResponse)
async def get_rag_stats():
    """
    Get RAG engine statistics
    """
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index documents into RAG system

    Supports: PDF, TXT, MD, DOCX
    Max size: 10MB per file
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []
    saved_paths = []

    try:
        rag_engine = get_rag_engine()

        # Process each file
        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            # Read and validate file size
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                logger.warning(f"File too large: {file.filename}")
                continue

            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = UPLOAD_DIR / safe_filename

            with open(file_path, "wb") as f:
                f.write(content)

            saved_paths.append(str(file_path))

            # Create document info
            doc_info = DocumentInfo(
                id=safe_filename,
                filename=file.filename,
                size=len(content),
                upload_date=datetime.now().isoformat(),
                file_type=file_ext,
            )
            uploaded_files.append(doc_info)

            logger.info(f"Saved file: {safe_filename}")

        if not saved_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid files to index. Check file type and size.",
            )

        # Index documents
        metadata = {
            "upload_date": datetime.now().isoformat(),
            "source": "api_upload",
        }

        result = rag_engine.index_documents(saved_paths, metadata=metadata)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])

        return UploadResponse(
            success=True,
            message=result["message"],
            indexed_count=result["indexed_count"],
            chunk_count=result.get("chunk_count", 0),
            files=uploaded_files,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Cleanup on error
        for path in saved_paths:
            try:
                os.remove(path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Query RAG system for relevant context

    Returns relevant document chunks with similarity scores
    """
    try:
        rag_engine = get_rag_engine()

        result = rag_engine.query(
            query_text=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        return RAGQueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all uploaded documents
    """
    try:
        documents = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                # Extract original filename (remove timestamp prefix)
                parts = file_path.name.split("_", 2)
                original_name = parts[2] if len(parts) > 2 else file_path.name

                doc_info = DocumentInfo(
                    id=file_path.name,
                    filename=original_name,
                    size=stat.st_size,
                    upload_date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    file_type=file_path.suffix,
                )
                documents.append(doc_info)

        return sorted(documents, key=lambda x: x.upload_date, reverse=True)

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document by ID
    """
    try:
        file_path = UPLOAD_DIR / doc_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete file
        os.remove(file_path)

        # Note: ChromaDB doesn't easily support deleting by filename
        # In production, you'd maintain a separate metadata DB to track doc_id -> chroma_id mapping
        logger.info(f"Deleted document: {doc_id}")

        return {
            "success": True,
            "message": f"Document {doc_id} deleted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_all_documents():
    """
    Clear all documents and reset RAG index

    WARNING: This will delete all indexed documents and files
    """
    try:
        rag_engine = get_rag_engine()

        # Clear ChromaDB collection
        success = rag_engine.clear_collection()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear collection")

        # Delete all uploaded files
        deleted_count = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                os.remove(file_path)
                deleted_count += 1

        logger.warning(f"Cleared all documents. Deleted {deleted_count} files.")

        return {
            "success": True,
            "message": f"Cleared RAG index and deleted {deleted_count} files",
            "deleted_count": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_all_documents():
    """
    Reindex all documents in the upload directory

    Useful after clearing the collection or when index is corrupted
    """
    try:
        rag_engine = get_rag_engine()

        # Get all document files
        file_paths = [
            str(p)
            for p in UPLOAD_DIR.glob("*")
            if p.is_file() and p.suffix in ALLOWED_EXTENSIONS
        ]

        if not file_paths:
            return {
                "success": True,
                "message": "No documents to reindex",
                "indexed_count": 0,
            }

        # Index all documents
        metadata = {
            "upload_date": datetime.now().isoformat(),
            "source": "reindex",
        }

        result = rag_engine.index_documents(file_paths, metadata=metadata)

        return {
            "success": result["success"],
            "message": result["message"],
            "indexed_count": result["indexed_count"],
            "chunk_count": result.get("chunk_count", 0),
        }

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
