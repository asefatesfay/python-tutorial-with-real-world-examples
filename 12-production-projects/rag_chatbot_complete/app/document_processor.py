"""
Document processing utilities for RAG pipeline.

Handles document loading, chunking, and preparation for embedding generation.
Supports multiple file formats: PDF, TXT, MD, DOCX.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


# ============================================================================
# Document Models
# ============================================================================

class DocumentChunk(BaseModel):
    """
    A chunk of text from a document.
    
    Represents a piece of text that will be embedded and stored
    in the vector database.
    """
    
    content: str
    metadata: dict
    chunk_id: str
    
    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        chunk_index: int,
        total_chunks: int,
        **extra_metadata
    ) -> "DocumentChunk":
        """
        Create a document chunk with automatic ID generation.
        
        Args:
            content: Text content of the chunk
            source: Source document identifier
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks in document
            **extra_metadata: Additional metadata fields
            
        Returns:
            DocumentChunk instance
        """
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{source}_{chunk_index}_{content[:100]}".encode()
        ).hexdigest()
        
        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            **extra_metadata
        }
        
        return cls(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id
        )


# ============================================================================
# Text Chunking
# ============================================================================

class TextChunker:
    """
    Splits text into chunks with overlap.
    
    Uses character-based chunking with configurable size and overlap.
    Tries to split on sentence boundaries when possible for better context.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]\s+')
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Algorithm:
        1. Try to split on sentence boundaries when possible
        2. If no sentence boundary in range, split on word boundary
        3. If no word boundary, split at exact character position
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
            
        Example:
            ```python
            chunker = TextChunker(chunk_size=100, chunk_overlap=20)
            chunks = chunker.chunk_text("Long text...")
            ```
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is the last chunk, take everything
            if end >= text_length:
                chunks.append(text[start:].strip())
                break
            
            # Try to find sentence boundary
            chunk_text = text[start:end]
            sentence_match = None
            
            for match in self.sentence_endings.finditer(chunk_text):
                sentence_match = match
            
            if sentence_match:
                # Split at last sentence boundary
                split_pos = start + sentence_match.end()
            else:
                # No sentence boundary, try word boundary
                # Look for last space before end
                chunk_text = text[start:end]
                last_space = chunk_text.rfind(' ')
                
                if last_space != -1:
                    split_pos = start + last_space
                else:
                    # No word boundary, split at exact position
                    split_pos = end
            
            # Add chunk
            chunks.append(text[start:split_pos].strip())
            
            # Move start position with overlap
            start = split_pos - self.chunk_overlap
            
            # Ensure we make progress
            if start <= chunks[-1] if chunks else 0:
                start = split_pos
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    def chunk_document(
        self,
        text: str,
        source: str,
        **extra_metadata
    ) -> List[DocumentChunk]:
        """
        Chunk a document and create DocumentChunk objects.
        
        Args:
            text: Document text
            source: Source identifier
            **extra_metadata: Additional metadata
            
        Returns:
            List of DocumentChunk objects
            
        Example:
            ```python
            chunker = TextChunker()
            chunks = chunker.chunk_document(
                text="Document content...",
                source="doc.pdf",
                author="John Doe"
            )
            ```
        """
        text_chunks = self.chunk_text(text)
        total_chunks = len(text_chunks)
        
        return [
            DocumentChunk.create(
                content=chunk,
                source=source,
                chunk_index=i,
                total_chunks=total_chunks,
                **extra_metadata
            )
            for i, chunk in enumerate(text_chunks)
        ]


# ============================================================================
# File Loaders
# ============================================================================

class DocumentLoader:
    """
    Load documents from various file formats.
    
    Supports: TXT, MD, PDF, DOCX
    """
    
    @staticmethod
    def load_text_file(file_path: Path) -> str:
        """
        Load plain text file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """
        Load PDF file.
        
        Requires: PyPDF2 or similar library
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
            
        Note:
            In production, you might want to use:
            - PyPDF2: Simple, works for most PDFs
            - pdfplumber: Better for tables
            - pymupdf (fitz): Faster, more features
        """
        try:
            import PyPDF2
            
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            
            return '\n\n'.join(text)
        
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. "
                "Install it with: pip install PyPDF2"
            )
    
    @staticmethod
    def load_docx(file_path: Path) -> str:
        """
        Load DOCX file.
        
        Requires: python-docx library
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text]
            return '\n\n'.join(paragraphs)
        
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX support. "
                "Install it with: pip install python-docx"
            )
    
    def load_document(self, file_path: Path) -> str:
        """
        Load document from any supported format.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document text
            
        Raises:
            ValueError: If file format is not supported
            
        Example:
            ```python
            loader = DocumentLoader()
            text = loader.load_document(Path("document.pdf"))
            ```
        """
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md']:
            return self.load_text_file(file_path)
        elif suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .txt, .md, .pdf, .docx"
            )


# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """
    Complete document processing pipeline.
    
    Combines loading and chunking into a single interface.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_file(
        self,
        file_path: Path,
        **extra_metadata
    ) -> List[DocumentChunk]:
        """
        Process a file into document chunks.
        
        Args:
            file_path: Path to file
            **extra_metadata: Additional metadata
            
        Returns:
            List of DocumentChunk objects
            
        Example:
            ```python
            processor = DocumentProcessor()
            chunks = processor.process_file(
                Path("document.pdf"),
                author="Jane Doe",
                category="technical"
            )
            ```
        """
        # Load document
        text = self.loader.load_document(file_path)
        
        # Clean text
        text = self.clean_text(text)
        
        # Chunk document
        chunks = self.chunker.chunk_document(
            text=text,
            source=str(file_path.name),
            file_type=file_path.suffix,
            **extra_metadata
        )
        
        return chunks
    
    def process_text(
        self,
        text: str,
        source: str,
        **extra_metadata
    ) -> List[DocumentChunk]:
        """
        Process raw text into document chunks.
        
        Args:
            text: Text to process
            source: Source identifier
            **extra_metadata: Additional metadata
            
        Returns:
            List of DocumentChunk objects
            
        Example:
            ```python
            processor = DocumentProcessor()
            chunks = processor.process_text(
                text="Long text content...",
                source="user_input",
                timestamp="2024-01-15"
            )
            ```
        """
        # Clean text
        text = self.clean_text(text)
        
        # Chunk text
        chunks = self.chunker.chunk_document(
            text=text,
            source=source,
            **extra_metadata
        )
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Removes excessive whitespace, normalizes line breaks, etc.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# ============================================================================
# Production Considerations
# ============================================================================

"""
Document Processing Best Practices:

1. **Chunking Strategy**:
   - Chunk size: 500-1500 characters works well for most cases
   - Overlap: 10-20% of chunk size prevents context loss
   - Split on sentence boundaries when possible
   
2. **Text Cleaning**:
   - Remove excessive whitespace
   - Normalize encodings
   - Handle special characters
   - Remove page numbers, headers, footers (for PDFs)
   
3. **Metadata**:
   - Include source filename
   - Add page numbers (for PDFs)
   - Store file type
   - Add processing timestamp
   - Include custom metadata (author, category, etc.)
   
4. **Error Handling**:
   - Validate file formats before processing
   - Handle encoding errors gracefully
   - Provide meaningful error messages
   - Log processing failures
   
5. **Performance**:
   - Process large files in batches
   - Use async processing for multiple files
   - Cache processed documents
   - Monitor memory usage
   
6. **File Format Support**:
   - TXT/MD: Simple, fast, reliable
   - PDF: Most common, but can be tricky
   - DOCX: Good for formatted documents
   - HTML: Requires special handling
   
Real-World Tips:
- Test with actual documents from your domain
- Different document types may need different chunk sizes
- Monitor embedding costs (chunks × embedding cost)
- Consider semantic chunking for complex documents
- Use metadata for filtering and better retrieval
"""
