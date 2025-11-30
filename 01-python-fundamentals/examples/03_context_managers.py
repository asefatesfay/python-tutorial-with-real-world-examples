"""
Context Managers for Senior Engineers

Similar to Go's defer, but more powerful and structured.
Essential for managing resources in ML/AI applications.

Run: python 03_context_managers.py
"""

import time
from contextlib import contextmanager
from typing import Generator, Any
from pathlib import Path
import tempfile
import shutil


# ============================================================================
# 1. Basic Context Manager with __enter__ and __exit__
# ============================================================================

class Timer:
    """
    Context manager for timing code blocks.
    Similar to Go's defer for start/end timing.
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0
    
    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"⏱️  Starting: {self.name}")
        self.start_time = time.time()
        return self  # Returned value assigned to 'as' variable
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting 'with' block (even on exception).
        
        Args:
            exc_type: Exception type (None if no exception)
            exc_val: Exception value
            exc_tb: Exception traceback
        
        Returns:
            True to suppress exception, False to propagate
        """
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  Finished: {self.name} ({self.elapsed:.4f}s)")
        
        if exc_type is not None:
            print(f"❌ Exception occurred: {exc_val}")
        
        return False  # Don't suppress exceptions


# Usage
def demo_timer():
    with Timer("Embedding generation"):
        time.sleep(0.5)
        # Automatically timed
    
    # Can also capture the context manager
    with Timer("Vector search") as timer:
        time.sleep(0.3)
    print(f"Took {timer.elapsed:.4f}s")


# ============================================================================
# 2. Context Manager with @contextmanager Decorator (Simpler)
# ============================================================================

@contextmanager
def temporary_directory(prefix: str = "ml_temp_") -> Generator[Path, None, None]:
    """
    Creates temporary directory, yields it, then cleans up.
    Essential for ML model artifacts, data processing.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    print(f"📁 Created temporary directory: {temp_dir}")
    
    try:
        yield temp_dir  # Code in 'with' block runs here
    finally:
        # Cleanup always runs, even on exception
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"🗑️  Cleaned up: {temp_dir}")


def demo_temp_directory():
    with temporary_directory("embeddings_") as temp_dir:
        # Use temporary directory
        file_path = temp_dir / "embeddings.txt"
        file_path.write_text("Some embeddings data")
        print(f"✅ Wrote to: {file_path}")
        # Automatically cleaned up when block exits


# ============================================================================
# 3. Database Connection Context Manager (Go-like defer pattern)
# ============================================================================

class VectorStoreConnection:
    """
    Vector store connection with automatic cleanup.
    Similar to database connections in Go.
    """
    
    def __init__(self, url: str):
        self.url = url
        self.connected = False
    
    def __enter__(self):
        """Establish connection."""
        print(f"🔌 Connecting to vector store: {self.url}")
        self.connected = True
        time.sleep(0.1)  # Simulate connection
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection."""
        print(f"🔌 Disconnecting from: {self.url}")
        self.connected = False
        return False
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search vector store."""
        if not self.connected:
            raise RuntimeError("Not connected!")
        print(f"🔍 Searching for: {query}")
        return [{"id": i, "score": 0.9 - i*0.1} for i in range(top_k)]


def demo_vector_store():
    # Connection automatically managed
    with VectorStoreConnection("http://localhost:6333") as store:
        results = store.search("AI tutorials", top_k=3)
        print(f"Found {len(results)} results")
    # Automatically disconnected


# ============================================================================
# 4. Nested Context Managers
# ============================================================================

@contextmanager
def profile_section(section_name: str) -> Generator[dict, None, None]:
    """Profile a section of code."""
    metrics = {"name": section_name}
    start = time.time()
    
    try:
        yield metrics
    finally:
        metrics["duration"] = time.time() - start
        print(f"📊 {section_name}: {metrics['duration']:.4f}s")


def demo_nested_contexts():
    """Multiple context managers in one statement."""
    with (
        temporary_directory("rag_") as temp_dir,
        VectorStoreConnection("http://localhost:6333") as store,
        profile_section("RAG Query") as metrics
    ):
        # All three contexts are active
        query = "What is Python?"
        results = store.search(query)
        
        # Save results to temp file
        output = temp_dir / "results.txt"
        output.write_text(str(results))
        
        metrics["results_count"] = len(results)


# ============================================================================
# 5. Exception Handling in Context Managers
# ============================================================================

class RetryContext:
    """
    Context manager that retries the block on failure.
    Different approach than decorator - for specific blocks.
    """
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # No exception, continue normally
        
        self.attempt += 1
        
        if self.attempt < self.max_attempts:
            print(f"⚠️  Attempt {self.attempt} failed, retrying...")
            time.sleep(self.delay)
            return True  # Suppress exception, retry
        else:
            print(f"❌ All {self.max_attempts} attempts failed")
            return False  # Propagate exception


# Note: This pattern requires a loop to actually retry
def demo_retry_context():
    for _ in range(3):  # Outer loop for retries
        with RetryContext(max_attempts=3, delay=0.5) as retry:
            # Simulate flaky operation
            import random
            if random.random() < 0.7:
                raise ConnectionError("API unavailable")
            print("✅ Operation succeeded!")
            break


# ============================================================================
# 6. Real-World Example: Model Loading Context
# ============================================================================

class ModelLoader:
    """
    Load ML model with automatic GPU memory management.
    Essential for managing scarce GPU resources.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def __enter__(self):
        """Load model to device."""
        print(f"🚀 Loading model '{self.model_name}' to {self.device}")
        time.sleep(0.5)  # Simulate loading
        self.model = {"name": self.model_name, "loaded": True}
        print(f"✅ Model loaded: {self.model_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unload model from device."""
        print(f"💾 Unloading model '{self.model_name}' from {self.device}")
        self.model = None
        print(f"✅ GPU memory freed")
        return False
    
    def predict(self, text: str) -> list[float]:
        """Run inference."""
        if not self.model:
            raise RuntimeError("Model not loaded!")
        print(f"🤖 Predicting for: {text[:30]}")
        return [0.1, 0.2, 0.3]


def demo_model_loader():
    # Model automatically loaded and unloaded
    with ModelLoader("sentence-transformers/all-MiniLM-L6-v2") as model:
        embedding1 = model.predict("Hello world")
        embedding2 = model.predict("AI is amazing")
        print(f"Generated {len([embedding1, embedding2])} embeddings")
    # GPU memory automatically freed


# ============================================================================
# 7. Context Manager for API Rate Limiting
# ============================================================================

@contextmanager
def rate_limited_session(calls_per_minute: int) -> Generator[dict, None, None]:
    """
    Context for rate-limited API calls.
    Tracks calls within the session.
    """
    session = {
        "calls": [],
        "limit": calls_per_minute,
        "start_time": time.time()
    }
    
    def check_rate_limit():
        now = time.time()
        # Remove calls older than 1 minute
        session["calls"] = [t for t in session["calls"] if now - t < 60]
        
        if len(session["calls"]) >= session["limit"]:
            sleep_time = 60 - (now - session["calls"][0])
            print(f"⏸️  Rate limit reached, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
            session["calls"] = []
    
    session["check"] = check_rate_limit
    
    try:
        yield session
    finally:
        total_calls = len(session["calls"])
        duration = time.time() - session["start_time"]
        print(f"📊 Session stats: {total_calls} calls in {duration:.1f}s")


def demo_rate_limiting():
    with rate_limited_session(calls_per_minute=10) as session:
        for i in range(5):
            session["check"]()
            print(f"📡 API call {i+1}")
            session["calls"].append(time.time())
            time.sleep(0.1)


# ============================================================================
# 8. Complete RAG Pipeline with Context Managers
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline using context managers."""
    
    @contextmanager
    def pipeline_context(
        self, 
        query: str
    ) -> Generator[dict, None, None]:
        """
        Complete pipeline context with:
        - Temporary storage
        - Vector store connection
        - Model loading
        - Performance profiling
        """
        pipeline_state = {
            "query": query,
            "start_time": time.time()
        }
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting RAG Pipeline for: '{query}'")
        print(f"{'='*60}\n")
        
        try:
            # Setup phase
            with (
                temporary_directory("rag_cache_") as temp_dir,
                VectorStoreConnection("http://localhost:6333") as vector_store,
                ModelLoader("gpt-4-embeddings") as model
            ):
                pipeline_state["temp_dir"] = temp_dir
                pipeline_state["vector_store"] = vector_store
                pipeline_state["model"] = model
                
                yield pipeline_state
                
        finally:
            # Cleanup and reporting
            duration = time.time() - pipeline_state["start_time"]
            print(f"\n{'='*60}")
            print(f"✅ Pipeline completed in {duration:.2f}s")
            print(f"{'='*60}\n")
    
    def execute(self, query: str) -> str:
        """Execute complete RAG pipeline."""
        with self.pipeline_context(query) as ctx:
            # Step 1: Embed query
            print("Step 1: Embedding query...")
            query_embedding = ctx["model"].predict(ctx["query"])
            
            # Step 2: Search vector store
            print("Step 2: Searching vector store...")
            results = ctx["vector_store"].search(ctx["query"], top_k=5)
            
            # Step 3: Cache results
            print("Step 3: Caching results...")
            cache_file = ctx["temp_dir"] / "cache.txt"
            cache_file.write_text(str(results))
            
            # Step 4: Generate response
            print("Step 4: Generating response...")
            response = f"Generated response based on {len(results)} documents"
            
            return response


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Context Manager Examples ===\n")
    
    print("1. Timer Context:")
    demo_timer()
    print()
    
    print("2. Temporary Directory:")
    demo_temp_directory()
    print()
    
    print("3. Vector Store Connection:")
    demo_vector_store()
    print()
    
    print("4. Nested Contexts:")
    demo_nested_contexts()
    print()
    
    print("5. Model Loader:")
    demo_model_loader()
    print()
    
    print("6. Rate Limited Session:")
    demo_rate_limiting()
    print()
    
    print("7. Complete RAG Pipeline:")
    pipeline = RAGPipeline()
    answer = pipeline.execute("What are context managers in Python?")
    print(f"📝 Final answer: {answer}")
    
    print("\n✅ Context managers ensure proper resource cleanup!")
    print("💡 Use for: files, connections, GPU memory, temp storage")


if __name__ == "__main__":
    main()
