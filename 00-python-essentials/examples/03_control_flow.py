"""
Control Flow - Python Essentials for Go/JS Engineers

Learn Python's control structures and list comprehensions.
Python's syntax is cleaner than Go's and more powerful than JS.

Run: poetry run python 00-python-essentials/examples/03_control_flow.py
"""


# ============================================================================
# 1. If/Elif/Else - Simpler than Go
# ============================================================================

def demo_if_statements():
    """
    Go:        if x > 0 { ... } else { ... }
    JS:        if (x > 0) { ... } else { ... }
    Python:    if x > 0: ... else: ...
    
    No parentheses needed, indentation matters!
    """
    print("=" * 70)
    print("1. If Statements - Clean Syntax")
    print("=" * 70)
    
    # Basic if
    age = 25
    if age >= 18:
        print(f"Age {age}: Adult")
    
    # If/else
    score = 85
    if score >= 90:
        print(f"Score {score}: A")
    else:
        print(f"Score {score}: B or lower")
    
    # If/elif/else chain
    temperature = 75
    if temperature > 80:
        status = "Hot"
    elif temperature > 60:
        status = "Nice"
    elif temperature > 40:
        status = "Cold"
    else:
        status = "Freezing"
    print(f"Temperature {temperature}°F: {status}")
    
    # Multiple conditions
    cpu_usage = 85
    memory_usage = 70
    
    if cpu_usage > 90 and memory_usage > 90:
        print("⚠️  Critical: Both CPU and memory high")
    elif cpu_usage > 80 or memory_usage > 80:
        print("⚠️  Warning: High resource usage")
    else:
        print("✅ System resources normal")
    
    # Ternary operator (like JS ? :)
    status = "active" if age >= 18 else "inactive"
    print(f"Status: {status}")
    
    print("\n💡 No parentheses, no braces - indentation is the syntax!")


# ============================================================================
# 2. Truthy/Falsy - Python's Rules
# ============================================================================

def demo_truthiness():
    """
    Python has specific rules for what's truthy/falsy.
    This is important for idiomatic Python!
    """
    print("\n" + "=" * 70)
    print("2. Truthy/Falsy Values")
    print("=" * 70)
    
    # Falsy values
    falsy_values = [
        (None, "None"),
        (False, "False"),
        (0, "0"),
        (0.0, "0.0"),
        ("", "empty string"),
        ([], "empty list"),
        ({}, "empty dict"),
        (set(), "empty set"),
    ]
    
    print("Falsy values:")
    for value, desc in falsy_values:
        if not value:
            print(f"  ✅ {desc}: falsy")
    
    # Truthy values (everything else!)
    truthy_values = [
        (True, "True"),
        (1, "1"),
        ("hello", "non-empty string"),
        ([1, 2], "non-empty list"),
        ({"a": 1}, "non-empty dict"),
    ]
    
    print("\nTruthy values:")
    for value, desc in truthy_values:
        if value:
            print(f"  ✅ {desc}: truthy")
    
    # Idiomatic checks
    items = []
    if items:  # Better than: if len(items) > 0
        print("Has items")
    else:
        print("✅ Empty list (idiomatic check)")
    
    user = None
    if user:  # Better than: if user is not None (for this case)
        print(f"User: {user}")
    else:
        print("✅ No user (idiomatic check)")
    
    print("\n💡 Use truthiness for cleaner code: 'if items:' not 'if len(items) > 0'")


# ============================================================================
# 3. For Loops - Much Cleaner than Go
# ============================================================================

def demo_for_loops():
    """
    Go:        for i := 0; i < len(items); i++ { ... }
    JS:        for (const item of items) { ... }
    Python:    for item in items: ...
    
    Python's for is simpler and more readable.
    """
    print("\n" + "=" * 70)
    print("3. For Loops - Clean Iteration")
    print("=" * 70)
    
    # Iterate over list
    languages = ["Python", "Go", "JavaScript"]
    print("Languages:")
    for lang in languages:
        print(f"  - {lang}")
    
    # Iterate with index (enumerate)
    print("\nWith index:")
    for i, lang in enumerate(languages, start=1):
        print(f"  {i}. {lang}")
    
    # Iterate over range (like Go's for i := 0; i < 5; i++)
    print("\nRange iteration:")
    for i in range(5):
        print(f"  {i}", end=" ")
    print()
    
    # Range with start and end
    print("\nRange(2, 7):")
    for i in range(2, 7):
        print(f"  {i}", end=" ")
    print()
    
    # Range with step
    print("\nEvens with range(0, 10, 2):")
    for i in range(0, 10, 2):
        print(f"  {i}", end=" ")
    print()
    
    # Iterate over dict
    user = {"name": "Alice", "age": 30, "city": "NYC"}
    print("\nDict iteration:")
    for key, value in user.items():
        print(f"  {key}: {value}")
    
    # Iterate over dict keys only
    print("\nKeys only:")
    for key in user.keys():  # or just: for key in user:
        print(f"  {key}")
    
    # Iterate over multiple lists (zip)
    names = ["Alice", "Bob", "Charlie"]
    ages = [30, 25, 35]
    print("\nZip iteration:")
    for name, age in zip(names, ages):
        print(f"  {name}: {age} years old")
    
    print("\n💡 Python's for loops are more readable than Go/JS!")


# ============================================================================
# 4. While Loops
# ============================================================================

def demo_while_loops():
    """
    While loops work similarly across languages.
    """
    print("\n" + "=" * 70)
    print("4. While Loops")
    print("=" * 70)
    
    # Basic while
    print("Countdown:")
    count = 5
    while count > 0:
        print(f"  {count}...", end=" ")
        count -= 1
    print("Go!")
    
    # While with break
    print("\nWith break:")
    count = 0
    while True:
        count += 1
        if count > 3:
            break
        print(f"  {count}", end=" ")
    print()
    
    # While with continue
    print("\nWith continue (skip 2):")
    count = 0
    while count < 5:
        count += 1
        if count == 2:
            continue
        print(f"  {count}", end=" ")
    print()
    
    print("\n💡 Use for loops when possible, while for unknown iterations!")


# ============================================================================
# 5. List Comprehensions - Powerful!
# ============================================================================

def demo_comprehensions():
    """
    List comprehensions are a Pythonic way to create lists.
    More concise and often faster than loops.
    """
    print("\n" + "=" * 70)
    print("5. List Comprehensions - Very Pythonic")
    print("=" * 70)
    
    # Basic comprehension
    squares = [x ** 2 for x in range(10)]
    print(f"Squares: {squares}")
    
    # With condition (filter)
    evens = [x for x in range(20) if x % 2 == 0]
    print(f"Evens: {evens}")
    
    # Transform strings
    names = ["alice", "bob", "charlie"]
    capitalized = [name.capitalize() for name in names]
    print(f"Capitalized: {capitalized}")
    
    # Filter and transform
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    even_squares = [x ** 2 for x in numbers if x % 2 == 0]
    print(f"Even squares: {even_squares}")
    
    # Nested comprehension (flatten)
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat = [num for row in matrix for num in row]
    print(f"Flattened: {flat}")
    
    # Dict comprehension
    squares_dict = {x: x ** 2 for x in range(5)}
    print(f"Dict: {squares_dict}")
    
    # Set comprehension
    unique_lengths = {len(word) for word in ["hi", "hello", "hey", "world"]}
    print(f"Set: {unique_lengths}")
    
    # Compare: for loop vs comprehension
    print("\nTraditional loop:")
    result = []
    for x in range(5):
        if x % 2 == 0:
            result.append(x ** 2)
    print(f"  {result}")
    
    print("Comprehension:")
    result = [x ** 2 for x in range(5) if x % 2 == 0]
    print(f"  {result}")
    
    print("\n💡 Use comprehensions for simple transformations - more Pythonic!")


# ============================================================================
# 6. Real-World: Log Filtering
# ============================================================================

def demo_log_filtering():
    """
    Real-world: Filter and analyze log entries.
    
    Common in DevOps:
    - Filter error logs
    - Extract specific patterns
    - Count occurrences
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Log Filtering")
    print("=" * 70)
    
    logs = [
        "2024-01-15 10:00:00 INFO User login successful",
        "2024-01-15 10:00:01 ERROR Database connection timeout",
        "2024-01-15 10:00:02 INFO Request processed in 45ms",
        "2024-01-15 10:00:03 ERROR API rate limit exceeded",
        "2024-01-15 10:00:04 WARN High memory usage: 85%",
        "2024-01-15 10:00:05 ERROR Database connection timeout",
        "2024-01-15 10:00:06 INFO User logout",
    ]
    
    print(f"Total logs: {len(logs)}")
    
    # Filter errors only
    errors = [log for log in logs if "ERROR" in log]
    print(f"\nErrors ({len(errors)}):")
    for error in errors:
        print(f"  {error}")
    
    # Filter by time range (after 10:00:02)
    recent = [log for log in logs if log.split()[1] > "10:00:02"]
    print(f"\nRecent logs ({len(recent)}):")
    for log in recent:
        print(f"  {log}")
    
    # Extract just the messages
    messages = [" ".join(log.split()[3:]) for log in logs]
    print(f"\nMessages:")
    for msg in messages:
        print(f"  - {msg}")
    
    # Count errors by type
    error_types = {}
    for log in errors:
        message = " ".join(log.split()[3:])
        error_types[message] = error_types.get(message, 0) + 1
    
    print(f"\nError breakdown:")
    for error, count in error_types.items():
        print(f"  {error}: {count}x")
    
    print("\n💡 Comprehensions make log filtering clean and readable!")


# ============================================================================
# 7. Real-World: Data Validation
# ============================================================================

def demo_data_validation():
    """
    Real-world: Validate user input or API data.
    
    Common scenarios:
    - Form validation
    - API request validation
    - Configuration validation
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Data Validation")
    print("=" * 70)
    
    # Validate user inputs
    users = [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "Bob", "age": -5, "email": "bob@invalid"},
        {"name": "", "age": 25, "email": "charlie@example.com"},
        {"name": "David", "age": 150, "email": "david@example.com"},
        {"name": "Eve", "age": 28, "email": "eve@example.com"},
    ]
    
    print(f"Validating {len(users)} users...")
    
    valid_users = []
    invalid_users = []
    
    for user in users:
        errors = []
        
        # Validate name
        if not user["name"]:
            errors.append("Name is empty")
        
        # Validate age
        if user["age"] < 0 or user["age"] > 120:
            errors.append(f"Age {user['age']} is invalid")
        
        # Validate email
        if "@" not in user["email"] or "." not in user["email"]:
            errors.append("Email format invalid")
        
        if errors:
            invalid_users.append({"user": user, "errors": errors})
        else:
            valid_users.append(user)
    
    print(f"\n✅ Valid users: {len(valid_users)}")
    for user in valid_users:
        print(f"  - {user['name']}")
    
    print(f"\n❌ Invalid users: {len(invalid_users)}")
    for item in invalid_users:
        user = item["user"]
        errors = item["errors"]
        print(f"  - {user['name'] or '(no name)'}: {', '.join(errors)}")
    
    print("\n💡 Validate early, fail fast, give clear error messages!")


# ============================================================================
# 8. Real-World: Batch Processing
# ============================================================================

def demo_batch_processing():
    """
    Real-world: Process items in batches.
    
    Common use cases:
    - API rate limiting (send 100 requests at a time)
    - Bulk database operations
    - Parallel processing chunks
    """
    print("\n" + "=" * 70)
    print("8. Real-World: Batch Processing")
    print("=" * 70)
    
    # Simulate large dataset
    document_ids = list(range(1, 251))  # 250 documents
    batch_size = 50
    
    print(f"Processing {len(document_ids)} documents in batches of {batch_size}")
    
    # Process in batches
    for i in range(0, len(document_ids), batch_size):
        batch = document_ids[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\nBatch {batch_num}:")
        print(f"  Processing docs {batch[0]} to {batch[-1]}")
        print(f"  Batch size: {len(batch)}")
        
        # Simulate processing
        # In real code: embed_documents(batch) or save_to_db(batch)
        success_count = len(batch)
        print(f"  ✅ Processed {success_count} documents")
    
    print("\n💡 Batching prevents overwhelming APIs and improves throughput!")


# ============================================================================
# 9. Real-World: Error Handling in Loops
# ============================================================================

def demo_error_handling_loops():
    """
    Real-world: Handle errors gracefully in loops.
    
    Common pattern: Continue processing even if some items fail.
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Error Handling in Loops")
    print("=" * 70)
    
    # Simulate API responses (some invalid)
    api_responses = [
        {"id": 1, "value": "42"},
        {"id": 2, "value": "invalid"},
        {"id": 3, "value": "100"},
        {"id": 4, "value": "not_a_number"},
        {"id": 5, "value": "75"},
    ]
    
    print(f"Processing {len(api_responses)} API responses...\n")
    
    successful = []
    failed = []
    
    for response in api_responses:
        try:
            # Try to convert value to int
            value = int(response["value"])
            successful.append({"id": response["id"], "value": value})
            print(f"✅ ID {response['id']}: {value}")
            
        except ValueError as e:
            failed.append({"id": response["id"], "error": str(e)})
            print(f"❌ ID {response['id']}: Failed to parse '{response['value']}'")
    
    print(f"\n📊 Results:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        total = sum(item["value"] for item in successful)
        print(f"  Total value: {total}")
    
    print("\n💡 Handle errors in loops to process as much as possible!")


# ============================================================================
# 10. Match Statement (Python 3.10+)
# ============================================================================

def demo_match_statement():
    """
    Python 3.10+ introduced match/case (like Go's switch).
    """
    print("\n" + "=" * 70)
    print("10. Match Statement (Python 3.10+)")
    print("=" * 70)
    
    def handle_status(status_code: int) -> str:
        match status_code:
            case 200:
                return "✅ OK"
            case 201:
                return "✅ Created"
            case 400:
                return "❌ Bad Request"
            case 401:
                return "❌ Unauthorized"
            case 404:
                return "❌ Not Found"
            case 500:
                return "❌ Server Error"
            case _:  # Default case
                return f"⚠️  Unknown status: {status_code}"
    
    # Test various status codes
    test_codes = [200, 404, 500, 999]
    print("HTTP status handling:")
    for code in test_codes:
        result = handle_status(code)
        print(f"  {code}: {result}")
    
    # Pattern matching with types
    def describe(value):
        match value:
            case int(x) if x > 0:
                return f"Positive integer: {x}"
            case int(x) if x < 0:
                return f"Negative integer: {x}"
            case str(s):
                return f"String: '{s}'"
            case list(items):
                return f"List with {len(items)} items"
            case _:
                return f"Something else: {type(value).__name__}"
    
    print("\nPattern matching:")
    test_values = [42, -10, "hello", [1, 2, 3], 3.14]
    for val in test_values:
        print(f"  {val} → {describe(val)}")
    
    print("\n💡 Match is powerful but if/elif works for most cases!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Control Flow for Go/JS Engineers\n")
    
    demo_if_statements()
    demo_truthiness()
    demo_for_loops()
    demo_while_loops()
    demo_comprehensions()
    demo_log_filtering()
    demo_data_validation()
    demo_batch_processing()
    demo_error_handling_loops()
    demo_match_statement()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. No parentheses in if statements: if x > 0:
2. Indentation is syntax - use 4 spaces
3. Use 'elif' not 'else if'
4. Learn truthiness: empty collections are falsy
5. For loops: for item in items:
6. Enumerate for index: for i, item in enumerate(items):
7. List comprehensions: [x**2 for x in nums if x > 0]
8. Use 'for' when you can, 'while' for unknown iterations
9. Match statement (3.10+) for complex branching
10. Handle errors in loops to keep processing

Pythonic patterns:
- if items:          # Better than: if len(items) > 0
- for item in items: # Better than: for i in range(len(items))
- [x*2 for x in nums] # Better than: loop + append
""")
    
    print("🎯 Next: 04_functions.py - Functions, arguments, returns")


if __name__ == "__main__":
    main()
