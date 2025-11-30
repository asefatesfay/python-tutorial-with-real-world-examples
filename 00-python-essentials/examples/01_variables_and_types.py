"""
Variables and Types - Python Essentials for Go/JS Engineers

Coming from Go with static typing, Python's dynamic typing can feel strange.
This example shows how Python's type system works in practice.

Run: poetry run python 00-python-essentials/examples/01_variables_and_types.py
"""


# ============================================================================
# 1. Variables - No Declaration Needed (Unlike Go)
# ============================================================================

def demo_variables():
    """
    Go requires:    var name string = "Alice"
    JS requires:    const name = "Alice"  // or let
    Python:         name = "Alice"        // That's it!
    """
    print("=" * 70)
    print("1. Variables - Simple Assignment")
    print("=" * 70)
    
    # Just assign - no var, const, or let
    name = "Alice"
    age = 30
    is_active = True
    
    print(f"name = {name} (type: {type(name).__name__})")
    print(f"age = {age} (type: {type(age).__name__})")
    print(f"is_active = {is_active} (type: {type(is_active).__name__})")
    
    # Variables can change type! (Unlike Go)
    age = "thirty"  # Was int, now str - totally valid!
    print(f"\nage changed to: {age} (type: {type(age).__name__})")
    
    print("\n💡 Python is dynamically typed - variables can change type!")


# ============================================================================
# 2. Basic Types - Similar to Go/JS but Dynamic
# ============================================================================

def demo_basic_types():
    """
    Python's basic types and how they compare to Go/JS.
    """
    print("\n" + "=" * 70)
    print("2. Basic Types")
    print("=" * 70)
    
    # Integers (like Go's int, JS's number)
    count = 42
    big_number = 1_000_000  # Underscores for readability!
    print(f"Integer: {count}, {big_number}")
    
    # Floats (like Go's float64, JS's number)
    price = 19.99
    scientific = 1.5e-3  # 0.0015
    print(f"Float: {price}, {scientific}")
    
    # Strings (like Go's string, JS's string)
    single = 'Hello'
    double = "World"
    multiline = """This is a
    multi-line string"""
    print(f"Strings: {single}, {double}")
    
    # Booleans (like Go's bool, JS's boolean)
    is_ready = True
    is_error = False
    print(f"Booleans: {is_ready}, {is_error}")
    
    # None (like Go's nil, JS's null/undefined)
    result = None
    print(f"None: {result}")
    
    print("\n💡 Most types work like you'd expect from Go/JS!")


# ============================================================================
# 3. Type Checking - Python's Duck Typing
# ============================================================================

def demo_duck_typing():
    """
    "If it walks like a duck and quacks like a duck, it's a duck"
    
    Go requires:    func process(x int) { ... }
    Python:         def process(x): ...  // Any type!
    """
    print("\n" + "=" * 70)
    print("3. Duck Typing - Flexibility")
    print("=" * 70)
    
    def describe(value):
        """This works with ANY type!"""
        return f"Value: {value}, Type: {type(value).__name__}"
    
    # Same function handles different types
    print(describe(42))
    print(describe("hello"))
    print(describe([1, 2, 3]))
    print(describe({"key": "value"}))
    
    print("\n💡 Functions can accept any type - powerful but needs care!")


# ============================================================================
# 4. Type Conversion - Common Operations
# ============================================================================

def demo_type_conversion():
    """
    Converting between types is common when working with APIs and data.
    """
    print("\n" + "=" * 70)
    print("4. Type Conversion")
    print("=" * 70)
    
    # String to int (like parsing user input, env vars)
    age_str = "25"
    age_int = int(age_str)
    print(f"String '{age_str}' → int {age_int}")
    
    # Int to string (like building messages)
    count = 100
    message = "Count: " + str(count)
    print(f"Int {count} → string: '{message}'")
    
    # String to float (like parsing prices, measurements)
    price_str = "19.99"
    price = float(price_str)
    print(f"String '{price_str}' → float {price}")
    
    # Float to int (truncates, doesn't round!)
    value = int(19.99)  # Becomes 19, not 20!
    print(f"Float 19.99 → int {value} (truncated, not rounded!)")
    
    # String to boolean - GOTCHA!
    # In Python, non-empty strings are True
    print(f"\nboolean conversions:")
    print(f"  bool('False') = {bool('False')}")  # True! (non-empty string)
    print(f"  bool('') = {bool('')}")            # False (empty string)
    print(f"  bool(0) = {bool(0)}")              # False
    print(f"  bool(1) = {bool(1)}")              # True
    
    print("\n⚠️  GOTCHA: bool('False') is True because string is non-empty!")


# ============================================================================
# 5. Real-World: Parsing API Responses
# ============================================================================

def demo_api_response_parsing():
    """
    Real-world: Parse and validate data from APIs.
    
    Common in DevOps:
    - Reading AWS API responses
    - Parsing GitHub API data
    - Processing monitoring alerts
    """
    print("\n" + "=" * 70)
    print("5. Real-World: Parsing API Responses")
    print("=" * 70)
    
    # Simulate API response (JSON is parsed to dict)
    api_response = {
        "instance_id": "i-1234567890abcdef0",
        "instance_type": "t2.micro",
        "cpu_usage": "45.5",  # Often comes as string!
        "memory_gb": "1",
        "is_running": "true",  # Boolean as string!
        "tags": ["web", "production"]
    }
    
    print("Raw API response:")
    print(f"  cpu_usage: {api_response['cpu_usage']} (type: {type(api_response['cpu_usage']).__name__})")
    
    # Parse and convert types
    instance_id = api_response["instance_id"]
    cpu_usage = float(api_response["cpu_usage"])
    memory_gb = int(api_response["memory_gb"])
    is_running = api_response["is_running"].lower() == "true"  # Safe conversion
    tags = api_response["tags"]
    
    print("\nParsed values:")
    print(f"  instance_id: {instance_id} (str)")
    print(f"  cpu_usage: {cpu_usage}% (float)")
    print(f"  memory_gb: {memory_gb}GB (int)")
    print(f"  is_running: {is_running} (bool)")
    print(f"  tags: {tags} (list)")
    
    # Now we can do math and logic
    if cpu_usage > 80:
        print("\n⚠️  High CPU usage!")
    elif cpu_usage > 50:
        print("\n✅ CPU usage is elevated but OK")
    else:
        print("\n✅ CPU usage is normal")
    
    print("\n💡 Always convert API strings to correct types for processing!")


# ============================================================================
# 6. Real-World: Environment Variables and Config
# ============================================================================

def demo_env_vars_config():
    """
    Real-world: Parse environment variables (always strings!).
    
    Common in DevOps:
    - Reading config from env vars
    - Container configuration
    - Feature flags
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Environment Variables")
    print("=" * 70)
    
    # Simulate environment variables (always strings!)
    env_vars = {
        "PORT": "8080",
        "DEBUG": "true",
        "MAX_CONNECTIONS": "100",
        "RATE_LIMIT": "10.5",
        "API_KEY": "sk-1234567890",
    }
    
    print("Raw environment variables (all strings):")
    for key, value in env_vars.items():
        print(f"  {key}: '{value}'")
    
    # Parse with proper types
    port = int(env_vars.get("PORT", "3000"))
    debug = env_vars.get("DEBUG", "false").lower() == "true"
    max_connections = int(env_vars.get("MAX_CONNECTIONS", "50"))
    rate_limit = float(env_vars.get("RATE_LIMIT", "5.0"))
    api_key = env_vars.get("API_KEY")
    
    print("\nParsed configuration:")
    print(f"  port: {port} (int)")
    print(f"  debug: {debug} (bool)")
    print(f"  max_connections: {max_connections} (int)")
    print(f"  rate_limit: {rate_limit} (float)")
    print(f"  api_key: {api_key} (str)")
    
    # Use in application logic
    if debug:
        print("\n🐛 Debug mode is enabled")
    
    print(f"🚀 Starting server on port {port}")
    print(f"⚡ Rate limit: {rate_limit} requests/second")
    
    print("\n💡 Always parse env vars - they're always strings!")


# ============================================================================
# 7. Common Gotchas Coming from Go/JS
# ============================================================================

def demo_gotchas():
    """
    Common mistakes when coming from Go or JavaScript.
    """
    print("\n" + "=" * 70)
    print("7. Common Gotchas")
    print("=" * 70)
    
    # Gotcha 1: Division always returns float
    print("\n❌ GOTCHA 1: Division returns float (unlike Go)")
    result = 10 / 2
    print(f"  10 / 2 = {result} (type: {type(result).__name__})")
    print(f"  In Go: 10 / 2 = 5 (int)")
    print(f"  Use // for integer division: 10 // 2 = {10 // 2}")
    
    # Gotcha 2: No ++ or -- operators
    print("\n❌ GOTCHA 2: No ++ or -- (unlike Go/JS)")
    count = 5
    # count++  # SyntaxError!
    count += 1  # Use this instead
    print(f"  Use 'count += 1' instead of 'count++'")
    
    # Gotcha 3: Truthy/falsy values
    print("\n❌ GOTCHA 3: Truthy/falsy differs from JS")
    print(f"  Empty list [] is falsy: {bool([])}")
    print(f"  Empty dict {{}} is falsy: {bool({})}")
    print(f"  Empty string '' is falsy: {bool('')}")
    print(f"  Zero 0 is falsy: {bool(0)}")
    print(f"  None is falsy: {bool(None)}")
    
    # Gotcha 4: Integer overflow doesn't exist!
    print("\n✅ BONUS: No integer overflow (unlike Go)")
    big = 10 ** 100  # 1 followed by 100 zeros!
    print(f"  Python handles arbitrarily large integers:")
    print(f"  10^100 = {big}")
    print(f"  In Go, this would overflow!")
    
    print("\n💡 Python has different behavior - know the gotchas!")


# ============================================================================
# 8. Type Hints - Optional but Recommended
# ============================================================================

def demo_type_hints():
    """
    Python 3.5+ added optional type hints (like TypeScript for JS).
    They're not enforced at runtime, but help with:
    - IDE autocomplete
    - Code documentation
    - Type checking tools (mypy)
    """
    print("\n" + "=" * 70)
    print("8. Type Hints (Optional but Good Practice)")
    print("=" * 70)
    
    # Without type hints (old way)
    def greet_old(name):
        return f"Hello, {name}!"
    
    # With type hints (modern way)
    def greet_new(name: str) -> str:
        return f"Hello, {name}!"
    
    # Type hints for variables
    age: int = 25
    price: float = 19.99
    is_active: bool = True
    username: str = "alice"
    
    print("Functions with type hints:")
    print(f"  def greet(name: str) -> str:")
    print(f"    return f'Hello, {{name}}!'")
    
    print("\nVariable type hints:")
    print(f"  age: int = 25")
    print(f"  price: float = 19.99")
    print(f"  is_active: bool = True")
    
    # Still dynamic - hints are just hints!
    age = "twenty-five"  # Still valid! Runtime doesn't enforce.
    print(f"\nType hints don't prevent: age = 'twenty-five'")
    print(f"But your IDE will warn you!")
    
    print("\n💡 Use type hints - they help catch bugs early!")


# ============================================================================
# 9. Real-World: Data Validation
# ============================================================================

def demo_data_validation():
    """
    Real-world: Validate and sanitize user input.
    
    Common in:
    - API request validation
    - Form processing
    - Configuration validation
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Data Validation")
    print("=" * 70)
    
    def validate_port(port_str: str) -> int | None:
        """
        Validate port number from string input.
        Returns int if valid, None if invalid.
        """
        try:
            port = int(port_str)
            if 1 <= port <= 65535:
                return port
            else:
                print(f"  ❌ Port {port} out of range (1-65535)")
                return None
        except ValueError:
            print(f"  ❌ Invalid port: '{port_str}' is not a number")
            return None
    
    # Test various inputs
    test_inputs = ["8080", "80", "99999", "abc", "-1", "3000"]
    
    print("Validating port numbers:")
    for input_str in test_inputs:
        result = validate_port(input_str)
        if result:
            print(f"  ✅ '{input_str}' → {result}")
    
    print("\n💡 Always validate user input and handle conversion errors!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Variables and Types for Go/JS Engineers\n")
    
    demo_variables()
    demo_basic_types()
    demo_duck_typing()
    demo_type_conversion()
    demo_api_response_parsing()
    demo_env_vars_config()
    demo_gotchas()
    demo_type_hints()
    demo_data_validation()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Python is dynamically typed - no type declarations needed
2. Variables can change type (unlike Go)
3. Use type hints for better code (like TypeScript)
4. Always convert API/env var strings to correct types
5. Watch out for gotchas: /, truthy values, no ++
6. Use isinstance() or type() to check types at runtime
7. Validate user input - never trust external data
8. Python integers can be arbitrarily large (no overflow!)
""")
    
    print("🎯 Next: 02_collections.py - Lists, dicts, sets, tuples")


if __name__ == "__main__":
    main()
