"""
Error Handling - Python Essentials for Go/JS Engineers

Learn Python's exception-based error handling.
Very different from Go's error returns, similar to JS try/catch.

Run: poetry run python 00-python-essentials/examples/05_error_handling.py
"""

import json
from typing import Any


# ============================================================================
# 1. Try/Except Basics - Like JS, Unlike Go
# ============================================================================

def demo_try_except():
    """
    Go:        if err != nil { return err }
    JS:        try { ... } catch (error) { ... }
    Python:    try: ... except Exception as e: ...
    
    Python uses exceptions, not error returns.
    """
    print("=" * 70)
    print("1. Try/Except Basics")
    print("=" * 70)
    
    # Basic try/except
    try:
        result = 10 / 2
        print(f"✅ Division successful: {result}")
    except ZeroDivisionError as e:
        print(f"❌ Error: {e}")
    
    # Catching errors
    try:
        result = 10 / 0  # Will raise ZeroDivisionError
        print(f"Result: {result}")
    except ZeroDivisionError as e:
        print(f"❌ Caught error: {e}")
    
    # Multiple except blocks
    def parse_and_divide(value_str: str, divisor_str: str):
        try:
            value = int(value_str)
            divisor = int(divisor_str)
            result = value / divisor
            return f"✅ Result: {result}"
        except ValueError as e:
            return f"❌ Invalid number: {e}"
        except ZeroDivisionError as e:
            return f"❌ Division error: {e}"
    
    print(f"\n{parse_and_divide('10', '2')}")
    print(f"{parse_and_divide('abc', '2')}")
    print(f"{parse_and_divide('10', '0')}")
    
    # Catch any exception
    try:
        data = json.loads("invalid json")
    except Exception as e:
        print(f"\n❌ Caught generic exception: {type(e).__name__}: {e}")
    
    print("\n💡 Python uses exceptions, not error returns like Go!")


# ============================================================================
# 2. Common Exception Types
# ============================================================================

def demo_exception_types():
    """
    Python has many built-in exception types.
    """
    print("\n" + "=" * 70)
    print("2. Common Exception Types")
    print("=" * 70)
    
    # ValueError - invalid value
    try:
        age = int("not_a_number")
    except ValueError as e:
        print(f"ValueError: {e}")
    
    # KeyError - missing dict key
    try:
        user = {"name": "Alice"}
        email = user["email"]  # Key doesn't exist
    except KeyError as e:
        print(f"KeyError: {e}")
    
    # IndexError - invalid list index
    try:
        items = [1, 2, 3]
        item = items[10]  # Index out of range
    except IndexError as e:
        print(f"IndexError: {e}")
    
    # TypeError - wrong type
    try:
        result = "hello" + 5  # Can't add str and int
    except TypeError as e:
        print(f"TypeError: {e}")
    
    # AttributeError - missing attribute
    try:
        x = "hello"
        x.append("world")  # Strings don't have append
    except AttributeError as e:
        print(f"AttributeError: {e}")
    
    # FileNotFoundError - file doesn't exist
    try:
        with open("nonexistent.txt") as f:
            content = f.read()
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    
    print("\n💡 Know the common exception types for better error handling!")


# ============================================================================
# 3. Finally Block - Always Executes
# ============================================================================

def demo_finally():
    """
    Finally block always executes, even if exception occurs.
    Use for cleanup (close files, connections, etc.)
    """
    print("\n" + "=" * 70)
    print("3. Finally Block")
    print("=" * 70)
    
    # Finally always runs
    def process_data(should_fail: bool):
        try:
            print("  1. Starting...")
            if should_fail:
                raise ValueError("Simulated error")
            print("  2. Processing...")
            return "success"
        except ValueError as e:
            print(f"  2. Error occurred: {e}")
            return "failed"
        finally:
            print("  3. Cleanup (finally block)")
    
    print("Success case:")
    result = process_data(False)
    print(f"Result: {result}\n")
    
    print("Error case:")
    result = process_data(True)
    print(f"Result: {result}")
    
    print("\n💡 Use finally for cleanup that must always happen!")


# ============================================================================
# 4. Else Block - Runs if No Exception
# ============================================================================

def demo_else():
    """
    Else block runs only if no exception occurred.
    """
    print("\n" + "=" * 70)
    print("4. Else Block")
    print("=" * 70)
    
    def safe_divide(a: float, b: float):
        try:
            result = a / b
        except ZeroDivisionError:
            print("  ❌ Cannot divide by zero")
        else:
            print(f"  ✅ Division successful: {result}")
        finally:
            print("  🧹 Cleanup done")
    
    print("Valid division:")
    safe_divide(10, 2)
    
    print("\nDivision by zero:")
    safe_divide(10, 0)
    
    print("\n💡 Else runs only if try succeeds - useful for flow control!")


# ============================================================================
# 5. Raising Exceptions - Creating Errors
# ============================================================================

def demo_raising_exceptions():
    """
    Raise exceptions when something goes wrong.
    """
    print("\n" + "=" * 70)
    print("5. Raising Exceptions")
    print("=" * 70)
    
    def validate_age(age: int) -> None:
        if age < 0:
            raise ValueError(f"Age cannot be negative: {age}")
        if age > 150:
            raise ValueError(f"Age too large: {age}")
        print(f"✅ Valid age: {age}")
    
    # Valid age
    try:
        validate_age(25)
    except ValueError as e:
        print(f"❌ {e}")
    
    # Invalid ages
    for invalid_age in [-5, 200]:
        try:
            validate_age(invalid_age)
        except ValueError as e:
            print(f"❌ {e}")
    
    # Re-raising exceptions
    def wrapper():
        try:
            validate_age(-10)
        except ValueError as e:
            print(f"Caught in wrapper: {e}")
            raise  # Re-raise the same exception
    
    print("\nRe-raising:")
    try:
        wrapper()
    except ValueError as e:
        print(f"Caught in main: {e}")
    
    print("\n💡 Raise exceptions for invalid states - fail fast!")


# ============================================================================
# 6. Custom Exceptions
# ============================================================================

def demo_custom_exceptions():
    """
    Create custom exception types for your application.
    """
    print("\n" + "=" * 70)
    print("6. Custom Exceptions")
    print("=" * 70)
    
    # Define custom exceptions
    class ValidationError(Exception):
        """Raised when validation fails."""
        pass
    
    class AuthenticationError(Exception):
        """Raised when authentication fails."""
        pass
    
    class RateLimitError(Exception):
        """Raised when rate limit exceeded."""
        def __init__(self, limit: int, retry_after: int):
            self.limit = limit
            self.retry_after = retry_after
            super().__init__(f"Rate limit {limit} exceeded. Retry after {retry_after}s")
    
    # Use custom exceptions
    def authenticate(token: str):
        if not token:
            raise AuthenticationError("Token is required")
        if token != "valid_token":
            raise AuthenticationError("Invalid token")
        return {"user_id": 123, "name": "Alice"}
    
    def make_api_call(count: int):
        if count > 100:
            raise RateLimitError(limit=100, retry_after=60)
        return {"status": "success"}
    
    # Test custom exceptions
    try:
        authenticate("")
    except AuthenticationError as e:
        print(f"❌ Auth error: {e}")
    
    try:
        make_api_call(150)
    except RateLimitError as e:
        print(f"❌ Rate limit: {e}")
        print(f"   Retry after: {e.retry_after}s")
    
    print("\n💡 Custom exceptions make error handling more semantic!")


# ============================================================================
# 7. Real-World: API Error Handling
# ============================================================================

def demo_api_error_handling():
    """
    Real-world: Handle various API errors gracefully.
    
    Common in:
    - Calling external APIs
    - OpenAI, AWS, GitHub API calls
    """
    print("\n" + "=" * 70)
    print("7. Real-World: API Error Handling")
    print("=" * 70)
    
    def call_api(endpoint: str, data: dict | None = None) -> dict:
        """
        Simulate API call with various error scenarios.
        """
        # Simulate different error conditions
        if "timeout" in endpoint:
            raise TimeoutError("Request timed out after 30s")
        if "auth" in endpoint:
            raise PermissionError("Invalid API key")
        if "notfound" in endpoint:
            raise FileNotFoundError("Endpoint not found")
        if "ratelimit" in endpoint:
            raise Exception("Rate limit exceeded")
        
        # Success
        return {"status": "success", "data": data}
    
    def safe_api_call(endpoint: str, data: dict | None = None) -> dict:
        """
        Make API call with comprehensive error handling.
        """
        try:
            result = call_api(endpoint, data)
            print(f"✅ Success: {endpoint}")
            return result
            
        except TimeoutError as e:
            print(f"⏱️  Timeout: {e}")
            return {"status": "error", "type": "timeout", "message": str(e)}
            
        except PermissionError as e:
            print(f"🔒 Auth error: {e}")
            return {"status": "error", "type": "auth", "message": str(e)}
            
        except FileNotFoundError as e:
            print(f"🔍 Not found: {e}")
            return {"status": "error", "type": "not_found", "message": str(e)}
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {"status": "error", "type": "unknown", "message": str(e)}
    
    # Test various endpoints
    endpoints = [
        "/users",
        "/timeout",
        "/auth",
        "/notfound",
        "/ratelimit",
    ]
    
    for endpoint in endpoints:
        result = safe_api_call(endpoint)
        print(f"  Result: {result}\n")
    
    print("💡 Always handle API errors - networks are unreliable!")


# ============================================================================
# 8. Real-World: Data Validation with Errors
# ============================================================================

def demo_validation_errors():
    """
    Real-world: Validate data and collect multiple errors.
    
    Common in:
    - Form validation
    - API request validation
    - Configuration validation
    """
    print("\n" + "=" * 70)
    print("8. Real-World: Data Validation")
    print("=" * 70)
    
    class ValidationError(Exception):
        """Custom validation error with multiple messages."""
        def __init__(self, errors: list[str]):
            self.errors = errors
            super().__init__(f"{len(errors)} validation error(s)")
    
    def validate_user(user: dict) -> dict:
        """
        Validate user data, collecting all errors.
        """
        errors = []
        
        # Validate name
        if not user.get("name"):
            errors.append("Name is required")
        elif len(user["name"]) < 2:
            errors.append("Name must be at least 2 characters")
        
        # Validate age
        age = user.get("age")
        if age is None:
            errors.append("Age is required")
        elif not isinstance(age, int):
            errors.append("Age must be an integer")
        elif age < 0 or age > 150:
            errors.append("Age must be between 0 and 150")
        
        # Validate email
        email = user.get("email", "")
        if not email:
            errors.append("Email is required")
        elif "@" not in email or "." not in email:
            errors.append("Email format is invalid")
        
        # If errors, raise exception
        if errors:
            raise ValidationError(errors)
        
        return user
    
    # Test cases
    test_users = [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "", "age": -5, "email": "invalid"},
        {"name": "B", "age": "thirty", "email": "bob@example.com"},
    ]
    
    for i, user in enumerate(test_users, 1):
        print(f"User {i}: {user}")
        try:
            validated = validate_user(user)
            print(f"  ✅ Valid\n")
        except ValidationError as e:
            print(f"  ❌ {len(e.errors)} error(s):")
            for error in e.errors:
                print(f"     - {error}")
            print()
    
    print("💡 Collect all validation errors at once for better UX!")


# ============================================================================
# 9. Real-World: Retry Logic with Errors
# ============================================================================

def demo_retry_logic():
    """
    Real-world: Retry failed operations with exponential backoff.
    
    Common for:
    - Network requests
    - Database connections
    - API calls
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Retry Logic")
    print("=" * 70)
    
    import time
    
    def unreliable_operation(attempt: int):
        """Simulate operation that fails sometimes."""
        if attempt < 3:
            raise ConnectionError(f"Network error (attempt {attempt})")
        return "success"
    
    def retry_with_backoff(
        operation,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ):
        """
        Retry operation with exponential backoff.
        """
        delay = initial_delay
        
        for attempt in range(1, max_retries + 1):
            try:
                result = operation(attempt)
                print(f"  ✅ Success on attempt {attempt}")
                return result
                
            except ConnectionError as e:
                if attempt == max_retries:
                    print(f"  ❌ Failed after {max_retries} attempts")
                    raise
                
                print(f"  ⚠️  Attempt {attempt} failed: {e}")
                print(f"  ⏳ Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    # Test retry logic
    try:
        result = retry_with_backoff(unreliable_operation, max_retries=5)
        print(f"Final result: {result}")
    except ConnectionError as e:
        print(f"Operation failed: {e}")
    
    print("\n💡 Retry with exponential backoff for transient failures!")


# ============================================================================
# 10. Error Handling Best Practices
# ============================================================================

def demo_best_practices():
    """
    Best practices for error handling in Python.
    """
    print("\n" + "=" * 70)
    print("10. Error Handling Best Practices")
    print("=" * 70)
    
    print("✅ DO:")
    print("  1. Catch specific exceptions, not generic Exception")
    print("  2. Use try/except/else/finally appropriately")
    print("  3. Create custom exceptions for your domain")
    print("  4. Include context in error messages")
    print("  5. Clean up resources in finally blocks")
    print("  6. Log errors before re-raising")
    print("  7. Validate input early (fail fast)")
    print("  8. Return error info, don't just print")
    
    print("\n❌ DON'T:")
    print("  1. Catch Exception without re-raising")
    print("  2. Use bare 'except:' (catches KeyboardInterrupt!)")
    print("  3. Ignore errors silently")
    print("  4. Put too much code in try blocks")
    print("  5. Use exceptions for control flow")
    
    # Good example
    print("\n✅ GOOD:")
    print('''
    def read_config(path: str) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config not found: {path}")
            return {}  # Return default
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            raise  # Re-raise for caller to handle
    ''')
    
    # Bad example
    print("❌ BAD:")
    print('''
    def read_config(path: str):
        try:
            with open(path) as f:
                return json.load(f)
        except:  # Too broad!
            pass  # Silent failure!
    ''')
    
    print("\n💡 Good error handling makes debugging much easier!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Error Handling for Go/JS Engineers\n")
    
    demo_try_except()
    demo_exception_types()
    demo_finally()
    demo_else()
    demo_raising_exceptions()
    demo_custom_exceptions()
    demo_api_error_handling()
    demo_validation_errors()
    demo_retry_logic()
    demo_best_practices()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. try/except - not error returns like Go
2. Multiple except blocks for different errors
3. finally - always executes (cleanup)
4. else - runs if no exception
5. raise - create exceptions
6. Custom exceptions - make errors semantic
7. Catch specific exceptions, not generic
8. Use exceptions for exceptional cases
9. Clean up resources in finally
10. Include context in error messages

Common exceptions:
- ValueError: Invalid value
- KeyError: Missing dict key
- IndexError: Invalid list index
- TypeError: Wrong type
- FileNotFoundError: File doesn't exist
- ConnectionError: Network issues

Python vs Go:
- Python: Exceptions (try/except)
- Go: Error returns (if err != nil)
- Python: Less boilerplate for error handling
- Go: More explicit error handling
""")
    
    print("🎯 Next: 06_files_and_io.py - File operations, JSON, CSV")


if __name__ == "__main__":
    main()
