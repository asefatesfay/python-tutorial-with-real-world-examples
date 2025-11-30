"""
Functions - Python Essentials for Go/JS Engineers

Learn Python's function syntax and powerful argument handling.
Python functions are more flexible than Go, similar to JS.

Run: poetry run python 00-python-essentials/examples/04_functions.py
"""

from typing import Any


# ============================================================================
# 1. Basic Functions - Simpler than Go
# ============================================================================

def demo_basic_functions():
    """
    Go:        func greet(name string) string { return ... }
    JS:        function greet(name) { return ... }
    Python:    def greet(name): return ...
    
    Python functions are flexible and dynamic.
    """
    print("=" * 70)
    print("1. Basic Functions")
    print("=" * 70)
    
    # Simple function
    def greet(name):
        return f"Hello, {name}!"
    
    # With type hints (recommended)
    def greet_typed(name: str) -> str:
        return f"Hello, {name}!"
    
    # Multiple parameters
    def add(a: int, b: int) -> int:
        return a + b
    
    # No return (returns None)
    def log_message(message: str) -> None:
        print(f"[LOG] {message}")
    
    # Test functions
    print(greet("Alice"))
    print(greet_typed("Bob"))
    print(f"add(5, 3) = {add(5, 3)}")
    log_message("System started")
    
    print("\n💡 Use type hints for better code documentation!")


# ============================================================================
# 2. Default Arguments - Very Useful
# ============================================================================

def demo_default_arguments():
    """
    Default arguments make functions more flexible.
    Like JS default params, better than Go's approach.
    """
    print("\n" + "=" * 70)
    print("2. Default Arguments")
    print("=" * 70)
    
    def connect(host: str = "localhost", port: int = 8080, timeout: int = 30):
        return f"Connecting to {host}:{port} (timeout: {timeout}s)"
    
    # Use defaults
    print(connect())
    
    # Override some defaults
    print(connect(host="api.example.com"))
    print(connect(port=3000))
    
    # Override all
    print(connect("db.example.com", 5432, 60))
    
    # Real-world: API client configuration
    def make_api_request(
        endpoint: str,
        method: str = "GET",
        timeout: int = 30,
        retry: int = 3
    ) -> dict:
        return {
            "endpoint": endpoint,
            "method": method,
            "timeout": timeout,
            "retry": retry
        }
    
    # Most parameters use defaults
    request1 = make_api_request("/users")
    print(f"\nAPI request 1: {request1}")
    
    # Override specific params
    request2 = make_api_request("/users", method="POST", retry=5)
    print(f"API request 2: {request2}")
    
    print("\n💡 Default args reduce boilerplate code!")


# ============================================================================
# 3. Keyword Arguments - Very Pythonic
# ============================================================================

def demo_keyword_arguments():
    """
    Keyword arguments make code more readable and flexible.
    """
    print("\n" + "=" * 70)
    print("3. Keyword Arguments")
    print("=" * 70)
    
    def create_user(name: str, age: int, email: str, active: bool = True):
        return {
            "name": name,
            "age": age,
            "email": email,
            "active": active
        }
    
    # Positional arguments
    user1 = create_user("Alice", 30, "alice@example.com")
    print(f"Positional: {user1}")
    
    # Keyword arguments (more readable!)
    user2 = create_user(
        name="Bob",
        age=25,
        email="bob@example.com",
        active=False
    )
    print(f"Keyword: {user2}")
    
    # Mixed (positional must come first)
    user3 = create_user("Charlie", age=35, email="charlie@example.com")
    print(f"Mixed: {user3}")
    
    # Order doesn't matter with keywords
    user4 = create_user(
        email="david@example.com",
        name="David",
        age=28
    )
    print(f"Reordered: {user4}")
    
    print("\n💡 Use keyword args for functions with many parameters!")


# ============================================================================
# 4. *args and **kwargs - Variable Arguments
# ============================================================================

def demo_variable_arguments():
    """
    *args: Variable positional arguments (tuple)
    **kwargs: Variable keyword arguments (dict)
    
    Similar to JS's ...rest
    """
    print("\n" + "=" * 70)
    print("4. *args and **kwargs")
    print("=" * 70)
    
    # *args - accept any number of positional arguments
    def sum_all(*args: int) -> int:
        """Sum any number of integers."""
        return sum(args)
    
    print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
    print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")
    print(f"sum_all() = {sum_all()}")
    
    # **kwargs - accept any number of keyword arguments
    def print_config(**kwargs: Any) -> None:
        """Print configuration key-value pairs."""
        print("Configuration:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
    
    print_config(host="localhost", port=8080, debug=True)
    
    # Combining regular args, *args, and **kwargs
    def flexible_function(required, *args, default="value", **kwargs):
        print(f"  required: {required}")
        print(f"  args: {args}")
        print(f"  default: {default}")
        print(f"  kwargs: {kwargs}")
    
    print("\nFlexible function:")
    flexible_function(
        "must_provide",
        "extra1", "extra2",
        default="custom",
        option1="a",
        option2="b"
    )
    
    # Real-world: Wrapper function
    def api_call_with_retry(endpoint: str, *args, retry: int = 3, **kwargs):
        """Make API call with retry logic."""
        print(f"Calling {endpoint} (retry: {retry})")
        print(f"  Args: {args}")
        print(f"  Kwargs: {kwargs}")
    
    print("\nAPI wrapper:")
    api_call_with_retry("/users", "param1", "param2", retry=5, timeout=30, headers={"Auth": "token"})
    
    print("\n💡 *args and **kwargs make functions extremely flexible!")


# ============================================================================
# 5. Multiple Return Values - Cleaner than Go
# ============================================================================

def demo_multiple_returns():
    """
    Python returns tuples, which can be unpacked.
    Cleaner than Go's multiple returns!
    """
    print("\n" + "=" * 70)
    print("5. Multiple Return Values")
    print("=" * 70)
    
    def divide_with_remainder(a: int, b: int) -> tuple[int, int]:
        """Return quotient and remainder."""
        return a // b, a % b
    
    # Unpack return values
    quotient, remainder = divide_with_remainder(17, 5)
    print(f"17 ÷ 5 = {quotient} remainder {remainder}")
    
    # Real-world: Parse response with status
    def fetch_data(url: str) -> tuple[bool, dict | None, str | None]:
        """Fetch data, return (success, data, error)."""
        # Simulate API call
        if "valid" in url:
            return True, {"id": 1, "name": "Alice"}, None
        else:
            return False, None, "Invalid URL"
    
    # Success case
    success, data, error = fetch_data("https://api.valid.com/users/1")
    if success:
        print(f"\n✅ Success: {data}")
    else:
        print(f"❌ Error: {error}")
    
    # Error case
    success, data, error = fetch_data("https://api.invalid.com")
    if success:
        print(f"✅ Success: {data}")
    else:
        print(f"❌ Error: {error}")
    
    # Can ignore values with _
    success, _, _ = fetch_data("https://api.valid.com/users/1")
    print(f"\nJust checking success: {success}")
    
    print("\n💡 Multiple returns are cleaner than Go's approach!")


# ============================================================================
# 6. Lambda Functions - Like JS Arrow Functions
# ============================================================================

def demo_lambda_functions():
    """
    Lambdas are anonymous functions, great for simple operations.
    Similar to JS arrow functions.
    """
    print("\n" + "=" * 70)
    print("6. Lambda Functions")
    print("=" * 70)
    
    # Basic lambda
    square = lambda x: x ** 2
    print(f"square(5) = {square(5)}")
    
    # Lambda with multiple arguments
    add = lambda a, b: a + b
    print(f"add(3, 4) = {add(3, 4)}")
    
    # Use in sorting
    users = [
        {"name": "Charlie", "age": 35},
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    
    # Sort by age
    sorted_by_age = sorted(users, key=lambda u: u["age"])
    print(f"\nSorted by age: {[u['name'] for u in sorted_by_age]}")
    
    # Sort by name
    sorted_by_name = sorted(users, key=lambda u: u["name"])
    print(f"Sorted by name: {[u['name'] for u in sorted_by_name]}")
    
    # Use in filter
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"\nEvens: {evens}")
    
    # Use in map
    squared = list(map(lambda x: x ** 2, numbers))
    print(f"Squared: {squared}")
    
    print("\n💡 Lambdas are great for simple operations, use def for complex ones!")


# ============================================================================
# 7. Real-World: API Client Functions
# ============================================================================

def demo_api_client():
    """
    Real-world: Building an API client with various function patterns.
    """
    print("\n" + "=" * 70)
    print("7. Real-World: API Client")
    print("=" * 70)
    
    def make_request(
        endpoint: str,
        method: str = "GET",
        headers: dict | None = None,
        data: dict | None = None,
        timeout: int = 30
    ) -> dict:
        """Make HTTP request (simplified)."""
        if headers is None:
            headers = {}
        
        print(f"  {method} {endpoint}")
        print(f"  Timeout: {timeout}s")
        if headers:
            print(f"  Headers: {headers}")
        if data:
            print(f"  Data: {data}")
        
        # Simulate response
        return {
            "status": 200,
            "data": {"message": "Success"}
        }
    
    # Simple GET
    print("1. Simple GET:")
    response = make_request("/users")
    
    # GET with auth
    print("\n2. GET with auth:")
    response = make_request(
        "/users/me",
        headers={"Authorization": "Bearer token123"}
    )
    
    # POST with data
    print("\n3. POST with data:")
    response = make_request(
        "/users",
        method="POST",
        headers={"Content-Type": "application/json"},
        data={"name": "Alice", "email": "alice@example.com"}
    )
    
    print("\n💡 Default args and keyword args make APIs clean!")


# ============================================================================
# 8. Real-World: Data Processing Pipeline
# ============================================================================

def demo_data_pipeline():
    """
    Real-world: Chain functions for data processing.
    """
    print("\n" + "=" * 70)
    print("8. Real-World: Data Processing Pipeline")
    print("=" * 70)
    
    def load_data(source: str) -> list[dict]:
        """Load data from source."""
        print(f"1. Loading from {source}")
        return [
            {"id": 1, "value": "  Hello  ", "score": 85},
            {"id": 2, "value": "World", "score": 92},
            {"id": 3, "value": "Python", "score": 78},
        ]
    
    def clean_data(data: list[dict]) -> list[dict]:
        """Clean data - strip whitespace."""
        print("2. Cleaning data")
        return [
            {**item, "value": item["value"].strip()}
            for item in data
        ]
    
    def filter_data(data: list[dict], min_score: int = 80) -> list[dict]:
        """Filter by score."""
        print(f"3. Filtering (min_score: {min_score})")
        return [item for item in data if item["score"] >= min_score]
    
    def transform_data(data: list[dict]) -> list[dict]:
        """Transform to upper case."""
        print("4. Transforming data")
        return [
            {**item, "value": item["value"].upper()}
            for item in data
        ]
    
    # Execute pipeline
    data = load_data("database")
    data = clean_data(data)
    data = filter_data(data, min_score=80)
    data = transform_data(data)
    
    print(f"\n✅ Final result: {data}")
    
    print("\n💡 Break complex logic into simple, testable functions!")


# ============================================================================
# 9. Real-World: Configuration Builder
# ============================================================================

def demo_config_builder():
    """
    Real-world: Build configuration with flexible function arguments.
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Configuration Builder")
    print("=" * 70)
    
    def create_config(
        environment: str,
        *features: str,
        debug: bool = False,
        **overrides: Any
    ) -> dict:
        """Create application configuration."""
        config = {
            "environment": environment,
            "debug": debug,
            "features": list(features),
            "defaults": {
                "host": "localhost",
                "port": 8080,
                "workers": 4,
            }
        }
        
        # Apply overrides
        config["defaults"].update(overrides)
        
        return config
    
    # Development config
    dev_config = create_config(
        "development",
        "hot-reload", "verbose-logging",
        debug=True
    )
    print("Development config:")
    print(f"  Environment: {dev_config['environment']}")
    print(f"  Debug: {dev_config['debug']}")
    print(f"  Features: {dev_config['features']}")
    print(f"  Defaults: {dev_config['defaults']}")
    
    # Production config
    prod_config = create_config(
        "production",
        "metrics", "caching", "rate-limiting",
        debug=False,
        host="0.0.0.0",
        port=80,
        workers=16
    )
    print("\nProduction config:")
    print(f"  Environment: {prod_config['environment']}")
    print(f"  Debug: {prod_config['debug']}")
    print(f"  Features: {prod_config['features']}")
    print(f"  Defaults: {prod_config['defaults']}")
    
    print("\n💡 *args and **kwargs enable flexible configuration!")


# ============================================================================
# 10. Function Best Practices
# ============================================================================

def demo_best_practices():
    """
    Best practices for writing Python functions.
    """
    print("\n" + "=" * 70)
    print("10. Function Best Practices")
    print("=" * 70)
    
    # ✅ GOOD: Clear name, type hints, docstring
    def calculate_total_price(
        items: list[dict],
        tax_rate: float = 0.08
    ) -> float:
        """
        Calculate total price including tax.
        
        Args:
            items: List of items with 'price' key
            tax_rate: Tax rate as decimal (default: 0.08 = 8%)
            
        Returns:
            Total price including tax
        """
        subtotal = sum(item["price"] for item in items)
        return subtotal * (1 + tax_rate)
    
    # ❌ BAD: Mutable default argument
    def bad_append(item, items=[]):  # BUG! List shared across calls
        items.append(item)
        return items
    
    # ✅ GOOD: Use None for mutable defaults
    def good_append(item, items=None):
        if items is None:
            items = []
        items.append(item)
        return items
    
    # Test
    items = [{"price": 10.00}, {"price": 15.50}, {"price": 8.25}]
    total = calculate_total_price(items, tax_rate=0.08)
    print(f"✅ Total with tax: ${total:.2f}")
    
    # Show mutable default problem
    print(f"\n❌ Bad (mutable default):")
    print(f"  First call: {bad_append(1)}")
    print(f"  Second call: {bad_append(2)}")  # Contains 1 and 2!
    
    print(f"\n✅ Good (None default):")
    print(f"  First call: {good_append(1)}")
    print(f"  Second call: {good_append(2)}")  # Only contains 2
    
    print("\n💡 Best practices:")
    print("  1. Use type hints")
    print("  2. Write docstrings")
    print("  3. Clear, descriptive names")
    print("  4. Avoid mutable default arguments")
    print("  5. Keep functions focused (single responsibility)")
    print("  6. Use keyword args for clarity")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Functions for Go/JS Engineers\n")
    
    demo_basic_functions()
    demo_default_arguments()
    demo_keyword_arguments()
    demo_variable_arguments()
    demo_multiple_returns()
    demo_lambda_functions()
    demo_api_client()
    demo_data_pipeline()
    demo_config_builder()
    demo_best_practices()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. def name(params): - simple syntax
2. Type hints: def func(x: int) -> str:
3. Default args: def func(x=10):
4. Keyword args: func(name="Alice", age=30)
5. *args: variable positional arguments
6. **kwargs: variable keyword arguments
7. Multiple returns: return a, b, c
8. Lambda: lambda x: x ** 2
9. Always use None for mutable defaults
10. Write docstrings for complex functions

Function patterns:
- API clients: default args + keyword args
- Data pipelines: chain simple functions
- Config builders: *args + **kwargs
- Wrappers: *args + **kwargs to pass through

Avoid:
- ❌ Mutable defaults: def func(items=[])
- ❌ Too many parameters (use config dict)
- ❌ Complex logic in lambdas
""")
    
    print("🎯 Next: 05_error_handling.py - try/except, exceptions")


if __name__ == "__main__":
    main()
