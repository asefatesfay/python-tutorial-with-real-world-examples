"""
HTTP and APIs - Python Essentials for Go/JS Engineers

Learn how to make HTTP requests and work with APIs in Python.
Uses the 'requests' library - the de facto standard (like axios in JS).

Run: poetry run python 00-python-essentials/examples/08_http_and_apis.py
"""

import json
import time
from typing import Any, Dict, Optional


# Note: This example uses simulated responses for demonstration.
# In real code, install and use: pip install requests
# import requests


# ============================================================================
# Simulated HTTP Client for Demonstration
# ============================================================================

class SimulatedResponse:
    """Simulate HTTP response object (like requests.Response)."""
    
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)
        self.headers = {"Content-Type": "application/json"}
    
    def json(self) -> dict:
        """Parse JSON response."""
        return self._data
    
    def raise_for_status(self):
        """Raise exception for bad status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class SimulatedHTTP:
    """Simulated HTTP client (like requests library)."""
    
    @staticmethod
    def get(url: str, params: dict = None, headers: dict = None, timeout: int = 30) -> SimulatedResponse:
        """Simulate GET request."""
        print(f"  GET {url}")
        if params:
            print(f"    Params: {params}")
        if headers:
            print(f"    Headers: {headers}")
        
        # Simulate different responses
        if "users" in url:
            return SimulatedResponse(200, {"users": [{"id": 1, "name": "Alice"}]})
        elif "error" in url:
            return SimulatedResponse(404, {"error": "Not found"})
        else:
            return SimulatedResponse(200, {"message": "Success"})
    
    @staticmethod
    def post(url: str, json: dict = None, data: dict = None, headers: dict = None, timeout: int = 30) -> SimulatedResponse:
        """Simulate POST request."""
        print(f"  POST {url}")
        if json:
            print(f"    JSON: {json}")
        if data:
            print(f"    Data: {data}")
        if headers:
            print(f"    Headers: {headers}")
        
        return SimulatedResponse(201, {"id": 123, "status": "created"})
    
    @staticmethod
    def put(url: str, json: dict = None, headers: dict = None, timeout: int = 30) -> SimulatedResponse:
        """Simulate PUT request."""
        print(f"  PUT {url}")
        if json:
            print(f"    JSON: {json}")
        return SimulatedResponse(200, {"status": "updated"})
    
    @staticmethod
    def delete(url: str, headers: dict = None, timeout: int = 30) -> SimulatedResponse:
        """Simulate DELETE request."""
        print(f"  DELETE {url}")
        return SimulatedResponse(204, {})


# Use simulated client for demonstration
requests = SimulatedHTTP()


# ============================================================================
# 1. Basic GET Requests
# ============================================================================

def demo_basic_get():
    """
    Basic GET requests - reading data from APIs.
    
    Real code:
        import requests
        response = requests.get('https://api.example.com/users')
    """
    print("=" * 70)
    print("1. Basic GET Requests")
    print("=" * 70)
    
    # Simple GET request
    response = requests.get("https://api.example.com/users")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # GET with query parameters
    print("\nGET with parameters:")
    response = requests.get(
        "https://api.example.com/users",
        params={"page": 2, "limit": 50}
    )
    print(f"Status: {response.status_code}")
    
    # GET with headers
    print("\nGET with headers:")
    response = requests.get(
        "https://api.example.com/users",
        headers={
            "Authorization": "Bearer token123",
            "Accept": "application/json"
        }
    )
    print(f"Status: {response.status_code}")
    
    print("\n💡 requests.get() is like fetch() in JS or http.Get() in Go!")


# ============================================================================
# 2. POST Requests - Creating Data
# ============================================================================

def demo_post_requests():
    """
    POST requests - creating new resources.
    """
    print("\n" + "=" * 70)
    print("2. POST Requests")
    print("=" * 70)
    
    # POST with JSON data
    new_user = {
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30
    }
    
    response = requests.post(
        "https://api.example.com/users",
        json=new_user,  # Automatically sets Content-Type: application/json
        headers={"Authorization": "Bearer token123"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # POST with form data
    print("\nPOST with form data:")
    form_data = {
        "username": "alice",
        "password": "secret123"
    }
    
    response = requests.post(
        "https://api.example.com/login",
        data=form_data  # Sends as form-encoded data
    )
    print(f"Status: {response.status_code}")
    
    print("\n💡 Use json= for JSON, data= for form data!")


# ============================================================================
# 3. PUT and DELETE Requests
# ============================================================================

def demo_put_delete():
    """
    PUT (update) and DELETE requests.
    """
    print("\n" + "=" * 70)
    print("3. PUT and DELETE Requests")
    print("=" * 70)
    
    # PUT - update resource
    updated_user = {
        "name": "Alice Smith",
        "email": "alice.smith@example.com"
    }
    
    response = requests.put(
        "https://api.example.com/users/123",
        json=updated_user,
        headers={"Authorization": "Bearer token123"}
    )
    
    print(f"PUT status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # DELETE - remove resource
    print("\nDELETE request:")
    response = requests.delete(
        "https://api.example.com/users/123",
        headers={"Authorization": "Bearer token123"}
    )
    
    print(f"DELETE status: {response.status_code}")
    
    print("\n💡 PUT for updates, DELETE for removals!")


# ============================================================================
# 4. Error Handling
# ============================================================================

def demo_error_handling():
    """
    Handle HTTP errors gracefully.
    """
    print("\n" + "=" * 70)
    print("4. Error Handling")
    print("=" * 70)
    
    def safe_get(url: str) -> Optional[dict]:
        """Make GET request with error handling."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for 4xx/5xx
            return response.json()
            
        except Exception as e:
            # In real code, catch specific exceptions:
            # - requests.ConnectionError
            # - requests.Timeout
            # - requests.HTTPError
            print(f"  ❌ Error: {e}")
            return None
    
    # Success case
    print("Success case:")
    data = safe_get("https://api.example.com/users")
    if data:
        print(f"  ✅ Got data: {data}")
    
    # Error case
    print("\nError case:")
    data = safe_get("https://api.example.com/error")
    if not data:
        print(f"  ⚠️  Failed to fetch data")
    
    print("\n💡 Always handle network errors - APIs are unreliable!")


# ============================================================================
# 5. Real-World: API Client Class
# ============================================================================

def demo_api_client():
    """
    Real-world: Build a reusable API client class.
    
    Common pattern for:
    - OpenAI API
    - AWS APIs
    - GitHub API
    """
    print("\n" + "=" * 70)
    print("5. Real-World: API Client Class")
    print("=" * 70)
    
    class APIClient:
        """Reusable API client with common functionality."""
        
        def __init__(self, base_url: str, api_key: str):
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key
            self.session = None  # Would use requests.Session() in real code
        
        def _headers(self) -> dict:
            """Common headers for all requests."""
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MyApp/1.0"
            }
        
        def get(self, endpoint: str, params: dict = None) -> dict:
            """Make GET request."""
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, headers=self._headers())
            response.raise_for_status()
            return response.json()
        
        def post(self, endpoint: str, data: dict) -> dict:
            """Make POST request."""
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, headers=self._headers())
            response.raise_for_status()
            return response.json()
        
        def put(self, endpoint: str, data: dict) -> dict:
            """Make PUT request."""
            url = f"{self.base_url}{endpoint}"
            response = requests.put(url, json=data, headers=self._headers())
            response.raise_for_status()
            return response.json()
        
        def delete(self, endpoint: str) -> None:
            """Make DELETE request."""
            url = f"{self.base_url}{endpoint}"
            response = requests.delete(url, headers=self._headers())
            response.raise_for_status()
    
    # Use the client
    client = APIClient("https://api.example.com", "sk-test-key-123")
    
    print("Using API client:")
    
    # GET users
    users = client.get("/users", params={"page": 1})
    print(f"  GET /users: {users}")
    
    # POST new user
    new_user = client.post("/users", data={"name": "Alice", "email": "alice@example.com"})
    print(f"  POST /users: {new_user}")
    
    # PUT update
    updated = client.put("/users/123", data={"name": "Alice Smith"})
    print(f"  PUT /users/123: {updated}")
    
    print("\n💡 API client classes make code reusable and maintainable!")


# ============================================================================
# 6. Real-World: Retry Logic
# ============================================================================

def demo_retry_logic():
    """
    Real-world: Retry failed requests with exponential backoff.
    
    Essential for:
    - Network failures
    - Rate limits
    - Transient errors
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Retry Logic")
    print("=" * 70)
    
    def retry_request(
        url: str,
        max_retries: int = 3,
        backoff: float = 1.0
    ) -> Optional[dict]:
        """Make request with retry logic."""
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempt {attempt}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                print(f"  ✅ Success!")
                return response.json()
                
            except Exception as e:
                if attempt == max_retries:
                    print(f"  ❌ Failed after {max_retries} attempts")
                    return None
                
                wait_time = backoff * (2 ** (attempt - 1))
                print(f"  ⚠️  Failed: {e}")
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return None
    
    # Test retry logic
    print("Making request with retry:")
    data = retry_request("https://api.example.com/users", max_retries=3)
    
    print("\n💡 Retry with exponential backoff for resilient systems!")


# ============================================================================
# 7. Real-World: Rate Limiting
# ============================================================================

def demo_rate_limiting():
    """
    Real-world: Respect API rate limits.
    
    Common limits:
    - OpenAI: 3500 requests/minute
    - GitHub: 5000 requests/hour
    - AWS: varies by service
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Rate Limiting")
    print("=" * 70)
    
    class RateLimitedClient:
        """API client with rate limiting."""
        
        def __init__(self, base_url: str, api_key: str, rpm_limit: int = 60):
            self.base_url = base_url
            self.api_key = api_key
            self.rpm_limit = rpm_limit
            self.requests_made = 0
            self.window_start = time.time()
        
        def _check_rate_limit(self):
            """Check if we're within rate limit."""
            now = time.time()
            elapsed = now - self.window_start
            
            # Reset window after 60 seconds
            if elapsed >= 60:
                self.requests_made = 0
                self.window_start = now
                return
            
            # Check if we've hit limit
            if self.requests_made >= self.rpm_limit:
                wait_time = 60 - elapsed
                print(f"  ⏳ Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests_made = 0
                self.window_start = time.time()
        
        def get(self, endpoint: str) -> dict:
            """Make rate-limited GET request."""
            self._check_rate_limit()
            self.requests_made += 1
            
            print(f"  Request {self.requests_made}/{self.rpm_limit} in window")
            response = requests.get(f"{self.base_url}{endpoint}")
            return response.json()
    
    # Test rate limiting
    client = RateLimitedClient("https://api.example.com", "key123", rpm_limit=5)
    
    print("Making requests with rate limiting:")
    for i in range(3):
        data = client.get(f"/users/{i}")
        print(f"  ✅ Request {i+1} completed")
    
    print("\n💡 Rate limiting prevents API bans and errors!")


# ============================================================================
# 8. Real-World: Calling OpenAI API (Example)
# ============================================================================

def demo_openai_pattern():
    """
    Real-world: Pattern for calling OpenAI API.
    
    This is a common pattern you'll use in AI/ML work.
    """
    print("\n" + "=" * 70)
    print("8. Real-World: OpenAI API Pattern")
    print("=" * 70)
    
    class OpenAIClient:
        """Simplified OpenAI API client."""
        
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.base_url = "https://api.openai.com/v1"
        
        def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> list[float]:
            """Generate embedding for text."""
            print(f"  Creating embedding for: '{text[:50]}...'")
            
            # Real implementation:
            # response = requests.post(
            #     f"{self.base_url}/embeddings",
            #     headers={"Authorization": f"Bearer {self.api_key}"},
            #     json={"input": text, "model": model}
            # )
            # return response.json()["data"][0]["embedding"]
            
            # Simulated response
            return [0.1, 0.2, 0.3]  # Actual would be 1536 dimensions
        
        def chat_completion(self, messages: list[dict], model: str = "gpt-4") -> str:
            """Generate chat completion."""
            print(f"  Calling {model}...")
            print(f"  Messages: {len(messages)}")
            
            # Real implementation:
            # response = requests.post(
            #     f"{self.base_url}/chat/completions",
            #     headers={"Authorization": f"Bearer {self.api_key}"},
            #     json={"model": model, "messages": messages}
            # )
            # return response.json()["choices"][0]["message"]["content"]
            
            # Simulated response
            return "This is a simulated response from GPT-4."
    
    # Use OpenAI client
    client = OpenAIClient("sk-your-api-key")
    
    print("OpenAI API usage:")
    
    # Generate embedding
    embedding = client.create_embedding("Hello, world!")
    print(f"  ✅ Embedding: {embedding[:3]}... (length: {len(embedding)})")
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python in one sentence."}
    ]
    response = client.chat_completion(messages)
    print(f"  ✅ Response: {response}")
    
    print("\n💡 This pattern works for any AI/ML API!")


# ============================================================================
# 9. Real-World: Batch Processing
# ============================================================================

def demo_batch_processing():
    """
    Real-world: Process items in batches to respect rate limits.
    
    Common for:
    - Embedding generation
    - Bulk data processing
    - Database operations
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Batch Processing")
    print("=" * 70)
    
    def process_in_batches(
        items: list[str],
        batch_size: int = 10,
        delay: float = 1.0
    ) -> list[dict]:
        """Process items in batches with delay."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(items) + batch_size - 1) // batch_size
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
            
            # Process batch (simulated)
            for item in batch:
                # Real code: response = requests.post(..., json={"text": item})
                results.append({"item": item, "processed": True})
            
            print(f"  ✅ Batch {batch_num} completed")
            
            # Delay between batches
            if i + batch_size < len(items):
                print(f"  ⏳ Waiting {delay}s before next batch...")
                time.sleep(delay)
        
        return results
    
    # Test batch processing
    documents = [f"Document {i}" for i in range(1, 26)]  # 25 documents
    
    print(f"Processing {len(documents)} documents in batches of 10:")
    results = process_in_batches(documents, batch_size=10, delay=0.5)
    
    print(f"\n✅ Processed {len(results)} documents")
    
    print("\n💡 Batch processing prevents overwhelming APIs!")


# ============================================================================
# 10. Best Practices
# ============================================================================

def demo_best_practices():
    """
    Best practices for working with HTTP APIs.
    """
    print("\n" + "=" * 70)
    print("10. HTTP/API Best Practices")
    print("=" * 70)
    
    print("✅ DO:")
    print("  1. Use requests library (pip install requests)")
    print("  2. Set timeouts on all requests")
    print("  3. Handle errors gracefully")
    print("  4. Retry with exponential backoff")
    print("  5. Respect rate limits")
    print("  6. Use sessions for multiple requests")
    print("  7. Include User-Agent header")
    print("  8. Log requests for debugging")
    print("  9. Use environment variables for API keys")
    print("  10. Batch requests when possible")
    
    print("\n❌ DON'T:")
    print("  1. Hardcode API keys in code")
    print("  2. Ignore rate limits")
    print("  3. Skip error handling")
    print("  4. Make synchronous calls in loops")
    print("  5. Forget timeouts")
    
    # Good example
    print("\n✅ GOOD:")
    print('''
    import requests
    from os import getenv
    
    def call_api(endpoint: str) -> dict:
        try:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {getenv('API_KEY')}"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return None
    ''')
    
    # Bad example
    print("❌ BAD:")
    print('''
    import requests
    
    # Hardcoded key, no timeout, no error handling
    response = requests.get(
        "https://api.example.com/data",
        headers={"Authorization": "Bearer sk-1234"}
    )
    data = response.json()  # Will crash if request fails!
    ''')
    
    print("\n💡 Follow best practices for reliable API interactions!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python HTTP and APIs for Go/JS Engineers\n")
    print("Note: This example uses simulated responses for demonstration.")
    print("In real code, install: pip install requests\n")
    
    demo_basic_get()
    demo_post_requests()
    demo_put_delete()
    demo_error_handling()
    demo_api_client()
    demo_retry_logic()
    demo_rate_limiting()
    demo_openai_pattern()
    demo_batch_processing()
    demo_best_practices()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Install requests: pip install requests
2. GET: requests.get(url, params={...})
3. POST: requests.post(url, json={...})
4. PUT: requests.put(url, json={...})
5. DELETE: requests.delete(url)
6. Always set timeout: timeout=30
7. Handle errors: try/except with raise_for_status()
8. Create API client classes for reusability
9. Implement retry with exponential backoff
10. Respect rate limits

Real-world patterns:
- API client classes for organization
- Retry logic for reliability
- Rate limiting for compliance
- Batch processing for efficiency
- Error handling for robustness

Common APIs you'll use:
- OpenAI: Embeddings, chat completions
- Anthropic: Claude API
- AWS: Various services
- GitHub: Repository data
- Vector DBs: ChromaDB, Pinecone

Next steps:
- Learn async HTTP (aiohttp, httpx) for concurrency
- Implement connection pooling
- Add request/response logging
- Build retry decorators
""")
    
    print("🎯 Congratulations! You've completed Python Essentials!")
    print("    Next: Module 1 - Python Fundamentals (decorators, etc.)")


if __name__ == "__main__":
    main()
