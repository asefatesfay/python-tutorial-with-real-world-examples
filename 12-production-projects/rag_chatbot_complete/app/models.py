"""
Pydantic models for request/response validation.

All API endpoints use these models for automatic validation,
serialization, and documentation generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Chat Models
# ============================================================================

class ConversationMessage(BaseModel):
    """
    A single message in a conversation.
    
    Represents either a user query or assistant response
    in the conversation history.
    """
    
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message sender"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Content of the message"
    )
    
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When the message was created"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What are the key features of this product?",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ChatRequest(BaseModel):
    """
    Request model for chat endpoints.
    
    Includes the user's query and optional conversation history
    for context-aware responses.
    """
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User's question or query"
    )
    
    conversation_history: List[ConversationMessage] = Field(
        default=[],
        max_length=20,
        description="Previous messages in the conversation"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=4000,
        description="Maximum tokens in response (overrides default)"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Response temperature (overrides default)"
    )
    
    @field_validator("conversation_history")
    @classmethod
    def validate_history_alternates(
        cls, 
        v: List[ConversationMessage]
    ) -> List[ConversationMessage]:
        """
        Validate that conversation history alternates between user and assistant.
        
        This ensures the conversation follows a natural back-and-forth pattern.
        """
        if len(v) == 0:
            return v
            
        for i in range(len(v) - 1):
            current_role = v[i].role
            next_role = v[i + 1].role
            
            if current_role == next_role:
                raise ValueError(
                    f"Conversation history must alternate between user and assistant. "
                    f"Found consecutive {current_role} messages at positions {i} and {i+1}."
                )
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key features of this product?",
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "Tell me about this product",
                        "timestamp": "2024-01-15T10:25:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "This is an advanced RAG chatbot...",
                        "timestamp": "2024-01-15T10:25:05Z"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
        }


class SourceChunk(BaseModel):
    """
    A retrieved document chunk used as context.
    
    Represents a piece of text retrieved from the vector store
    that was used to generate the response.
    """
    
    content: str = Field(
        ...,
        description="Text content of the chunk"
    )
    
    metadata: Dict[str, Any] = Field(
        default={},
        description="Metadata about the chunk (source, page, etc.)"
    )
    
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity score (if available)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "The RAG chatbot uses vector embeddings...",
                "metadata": {
                    "source": "documentation.pdf",
                    "page": 5,
                    "section": "Architecture"
                },
                "similarity_score": 0.89
            }
        }


class ChatResponse(BaseModel):
    """
    Response model for non-streaming chat endpoint.
    
    Contains the generated response, sources used, and metadata.
    """
    
    response: str = Field(
        ...,
        description="Generated response to the query"
    )
    
    sources: List[SourceChunk] = Field(
        default=[],
        description="Source chunks used for generating the response"
    )
    
    model: str = Field(
        ...,
        description="Model used for generation"
    )
    
    tokens_used: int = Field(
        ...,
        ge=0,
        description="Total tokens used (prompt + completion)"
    )
    
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time taken to process the request"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the response was generated"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the documentation, the key features include...",
                "sources": [
                    {
                        "content": "Feature 1: Vector search...",
                        "metadata": {"source": "docs.pdf", "page": 5},
                        "similarity_score": 0.92
                    }
                ],
                "model": "gpt-4-turbo-preview",
                "tokens_used": 450,
                "processing_time_ms": 1234.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StreamChunk(BaseModel):
    """
    A chunk in a streaming response.
    
    Used for Server-Sent Events (SSE) streaming endpoint.
    """
    
    type: Literal["token", "sources", "done", "error"] = Field(
        ...,
        description="Type of the chunk"
    )
    
    content: Optional[str] = Field(
        default=None,
        description="Token content (for type='token')"
    )
    
    sources: Optional[List[SourceChunk]] = Field(
        default=None,
        description="Source chunks (for type='sources')"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message (for type='error')"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "token",
                    "content": "The"
                },
                {
                    "type": "sources",
                    "sources": [
                        {
                            "content": "...",
                            "metadata": {"source": "doc.pdf"}
                        }
                    ]
                },
                {
                    "type": "done"
                }
            ]
        }


# ============================================================================
# Document Models
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload endpoint.
    
    Returns information about the processed document.
    """
    
    document_id: str = Field(
        ...,
        description="Unique identifier for the document"
    )
    
    filename: str = Field(
        ...,
        description="Original filename"
    )
    
    chunks: int = Field(
        ...,
        ge=0,
        description="Number of chunks created"
    )
    
    status: Literal["processed", "processing", "failed"] = Field(
        ...,
        description="Processing status"
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Additional message or error details"
    )
    
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time taken to process the document"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the document was processed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123xyz",
                "filename": "product_documentation.pdf",
                "chunks": 42,
                "status": "processed",
                "message": "Document successfully processed",
                "processing_time_ms": 5678.9,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# Health Check Models
# ============================================================================

class DependencyHealth(BaseModel):
    """
    Health status of a dependency.
    
    Used in health check responses to report status of external services.
    """
    
    status: Literal["healthy", "unhealthy", "unknown"] = Field(
        ...,
        description="Health status"
    )
    
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Response latency in milliseconds"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if unhealthy"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "latency_ms": 45.2,
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Returns overall health status and dependency health.
    """
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status"
    )
    
    version: str = Field(
        ...,
        description="Application version"
    )
    
    dependencies: Dict[str, DependencyHealth] = Field(
        default={},
        description="Health status of dependencies"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "dependencies": {
                    "vector_store": {
                        "status": "healthy",
                        "latency_ms": 12.3
                    },
                    "openai": {
                        "status": "healthy",
                        "latency_ms": 234.5
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Response model for error responses.
    
    Provides consistent error reporting across all endpoints.
    """
    
    error: str = Field(
        ...,
        description="Error type or category"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details (debug only)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Query must be between 1 and 5000 characters",
                "detail": "Field: query, Value: ''",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# Production Considerations
# ============================================================================

"""
Why use Pydantic models?

1. **Automatic Validation**: FastAPI validates all requests automatically
   - Reject invalid requests before they reach your code
   - Save on API calls to OpenAI with bad inputs
   - Better error messages for clients
   
2. **Type Safety**: Get type hints in IDE
   - Fewer runtime errors
   - Better autocomplete
   - Easier refactoring
   
3. **Documentation**: Auto-generate OpenAPI/Swagger docs
   - Always up-to-date
   - Interactive API testing
   - Client SDK generation
   
4. **Serialization**: Automatic JSON conversion
   - datetime → ISO 8601 strings
   - Enums → strings
   - Nested models → nested JSON
   
5. **Validation**: Rich validation rules
   - String length limits (prevent abuse)
   - Numeric ranges (prevent invalid configs)
   - Custom validators (business logic)
   
Production Tips:
- Keep max_length conservative to prevent abuse
- Validate conversation_history to ensure quality
- Use field_validator for complex business logic
- Provide good examples in json_schema_extra
- Return consistent error responses
"""
