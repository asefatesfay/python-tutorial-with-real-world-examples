"""
Collections - Python Essentials for Go/JS Engineers

Learn Python's core data structures and when to use each one.
Coming from Go: slices, maps, and structs work differently here.
Coming from JS: arrays and objects have Python equivalents.

Run: poetry run python 00-python-essentials/examples/02_collections.py
"""

from typing import Any


# ============================================================================
# 1. Lists - Like Go Slices / JS Arrays
# ============================================================================

def demo_lists():
    """
    Go:        []string{"a", "b", "c"}
    JS:        ["a", "b", "c"]
    Python:    ["a", "b", "c"]
    
    Lists are ordered, mutable, and can contain mixed types.
    """
    print("=" * 70)
    print("1. Lists - Ordered Collections")
    print("=" * 70)
    
    # Creating lists
    numbers = [1, 2, 3, 4, 5]
    names = ["Alice", "Bob", "Charlie"]
    mixed = [1, "two", 3.0, True, None]  # Can mix types!
    
    print(f"numbers: {numbers}")
    print(f"names: {names}")
    print(f"mixed: {mixed}")
    
    # Accessing elements (like Go/JS)
    print(f"\nFirst element: {numbers[0]}")
    print(f"Last element: {numbers[-1]}")  # Negative indexing!
    print(f"Second to last: {numbers[-2]}")
    
    # Slicing (more powerful than Go/JS)
    print(f"\nFirst 3: {numbers[:3]}")      # [1, 2, 3]
    print(f"Last 3: {numbers[-3:]}")        # [3, 4, 5]
    print(f"Middle: {numbers[1:4]}")        # [2, 3, 4]
    print(f"Every 2nd: {numbers[::2]}")     # [1, 3, 5]
    print(f"Reversed: {numbers[::-1]}")     # [5, 4, 3, 2, 1]
    
    # Modifying lists
    numbers.append(6)           # Add to end
    numbers.insert(0, 0)        # Insert at index
    numbers.remove(3)           # Remove by value
    popped = numbers.pop()      # Remove and return last
    
    print(f"\nAfter modifications: {numbers}")
    print(f"Popped value: {popped}")
    
    # Common operations
    print(f"\nLength: {len(numbers)}")
    print(f"Sum: {sum(numbers)}")
    print(f"Max: {max(numbers)}")
    print(f"Min: {min(numbers)}")
    print(f"Contains 5? {5 in numbers}")
    
    print("\n💡 Lists are like Go slices but more flexible!")


# ============================================================================
# 2. Dictionaries - Like Go Maps / JS Objects
# ============================================================================

def demo_dictionaries():
    """
    Go:        map[string]int{"age": 30}
    JS:        {age: 30} or {"age": 30}
    Python:    {"age": 30}
    
    Dicts are key-value pairs, unordered (before 3.7), mutable.
    """
    print("\n" + "=" * 70)
    print("2. Dictionaries - Key-Value Pairs")
    print("=" * 70)
    
    # Creating dicts
    user = {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "is_active": True
    }
    
    print(f"user: {user}")
    
    # Accessing values
    print(f"\nName: {user['name']}")
    print(f"Age: {user['age']}")
    
    # Safe access with get() - won't error if key missing
    print(f"Phone: {user.get('phone', 'N/A')}")  # Returns default
    
    # Adding/modifying
    user["phone"] = "555-1234"
    user["age"] = 31
    print(f"\nUpdated: {user}")
    
    # Removing
    del user["phone"]
    email = user.pop("email")  # Remove and return
    print(f"\nRemoved email: {email}")
    print(f"After removal: {user}")
    
    # Checking existence
    print(f"\n'name' exists? {'name' in user}")
    print(f"'phone' exists? {'phone' in user}")
    
    # Iterating
    print("\nIterating over dict:")
    for key, value in user.items():
        print(f"  {key}: {value}")
    
    # Keys and values
    print(f"\nKeys: {list(user.keys())}")
    print(f"Values: {list(user.values())}")
    
    print("\n💡 Dicts are like Go maps but maintain insertion order (3.7+)!")


# ============================================================================
# 3. Sets - Unique Collections
# ============================================================================

def demo_sets():
    """
    Go:        map[string]bool (for set behavior)
    JS:        new Set([1, 2, 3])
    Python:    {1, 2, 3}
    
    Sets are unordered, mutable, no duplicates.
    """
    print("\n" + "=" * 70)
    print("3. Sets - Unique Collections")
    print("=" * 70)
    
    # Creating sets
    tags = {"python", "golang", "javascript"}
    numbers = {1, 2, 3, 3, 3}  # Duplicates removed!
    
    print(f"tags: {tags}")
    print(f"numbers: {numbers}")  # Only {1, 2, 3}
    
    # Adding/removing
    tags.add("rust")
    tags.remove("golang")
    print(f"\nModified tags: {tags}")
    
    # Set operations
    frontend = {"javascript", "typescript", "react"}
    backend = {"python", "golang", "rust"}
    fullstack = {"python", "javascript"}
    
    print(f"\nfrontend: {frontend}")
    print(f"backend: {backend}")
    print(f"fullstack: {fullstack}")
    
    # Union (all languages)
    all_langs = frontend | backend
    print(f"\nUnion (|): {all_langs}")
    
    # Intersection (common languages)
    common = frontend & fullstack
    print(f"Intersection (&): {common}")
    
    # Difference (only frontend)
    only_frontend = frontend - fullstack
    print(f"Difference (-): {only_frontend}")
    
    # Membership test (very fast!)
    print(f"\n'python' in backend? {'python' in backend}")
    
    print("\n💡 Sets are perfect for uniqueness and membership testing!")


# ============================================================================
# 4. Tuples - Immutable Sequences
# ============================================================================

def demo_tuples():
    """
    Go:        No direct equivalent (use structs)
    JS:        No direct equivalent (use arrays)
    Python:    (1, 2, 3)
    
    Tuples are ordered, immutable, often used for fixed data.
    """
    print("\n" + "=" * 70)
    print("4. Tuples - Immutable Sequences")
    print("=" * 70)
    
    # Creating tuples
    point = (10, 20)
    rgb = (255, 128, 0)
    single = (42,)  # Note the comma!
    
    print(f"point: {point}")
    print(f"rgb: {rgb}")
    print(f"single: {single}")
    
    # Accessing (like lists)
    print(f"\npoint x: {point[0]}, y: {point[1]}")
    
    # Unpacking (very Pythonic!)
    x, y = point
    r, g, b = rgb
    print(f"\nUnpacked: x={x}, y={y}")
    print(f"Unpacked: r={r}, g={g}, b={b}")
    
    # Immutable - can't modify!
    try:
        point[0] = 15
    except TypeError as e:
        print(f"\n❌ Can't modify tuple: {e}")
    
    # Multiple return values (returns a tuple)
    def get_user():
        return "Alice", 30, "alice@example.com"
    
    name, age, email = get_user()
    print(f"\nMultiple returns: {name}, {age}, {email}")
    
    # Use as dict keys (lists can't!)
    locations = {
        (0, 0): "origin",
        (10, 20): "point A",
        (30, 40): "point B"
    }
    print(f"\nTuples as keys: {locations[(10, 20)]}")
    
    print("\n💡 Use tuples for fixed data and multiple returns!")


# ============================================================================
# 5. Real-World: Processing Log Files
# ============================================================================

def demo_log_processing():
    """
    Real-world: Parse and analyze log entries.
    
    Common in DevOps:
    - Analyzing application logs
    - Finding error patterns
    - Counting occurrences
    """
    print("\n" + "=" * 70)
    print("5. Real-World: Processing Log Files")
    print("=" * 70)
    
    # Simulate log entries
    log_entries = [
        "2024-01-15 10:23:45 ERROR Database connection failed",
        "2024-01-15 10:23:46 INFO Request processed successfully",
        "2024-01-15 10:23:47 ERROR API timeout after 30s",
        "2024-01-15 10:23:48 WARN High memory usage: 85%",
        "2024-01-15 10:23:49 ERROR Database connection failed",
        "2024-01-15 10:23:50 INFO Request processed successfully",
    ]
    
    # Parse logs
    errors = []
    warnings = []
    info = []
    
    for entry in log_entries:
        if "ERROR" in entry:
            errors.append(entry)
        elif "WARN" in entry:
            warnings.append(entry)
        elif "INFO" in entry:
            info.append(entry)
    
    print(f"Total entries: {len(log_entries)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Info: {len(info)}")
    
    # Count error types (dict as counter)
    error_counts = {}
    for entry in errors:
        # Extract error message
        parts = entry.split("ERROR ", 1)
        if len(parts) > 1:
            error_msg = parts[1]
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
    
    print("\nError breakdown:")
    for error, count in error_counts.items():
        print(f"  {error}: {count}x")
    
    print("\n💡 Lists for sequences, dicts for counting - perfect combo!")


# ============================================================================
# 6. Real-World: API Response Processing
# ============================================================================

def demo_api_responses():
    """
    Real-world: Process data from REST APIs.
    
    Common tasks:
    - Extract relevant fields
    - Filter by criteria
    - Transform data format
    """
    print("\n" + "=" * 70)
    print("6. Real-World: API Response Processing")
    print("=" * 70)
    
    # Simulate API response (list of dicts)
    instances = [
        {"id": "i-001", "type": "t2.micro", "state": "running", "cpu": 45},
        {"id": "i-002", "type": "t2.small", "state": "running", "cpu": 78},
        {"id": "i-003", "type": "t2.micro", "state": "stopped", "cpu": 0},
        {"id": "i-004", "type": "t2.medium", "state": "running", "cpu": 92},
        {"id": "i-005", "type": "t2.micro", "state": "running", "cpu": 23},
    ]
    
    print(f"Total instances: {len(instances)}")
    
    # Filter running instances
    running = [inst for inst in instances if inst["state"] == "running"]
    print(f"Running instances: {len(running)}")
    
    # Find high CPU instances (>80%)
    high_cpu = [inst for inst in instances if inst["cpu"] > 80]
    print(f"\nHigh CPU instances:")
    for inst in high_cpu:
        print(f"  {inst['id']}: {inst['cpu']}% on {inst['type']}")
    
    # Count by instance type
    type_counts = {}
    for inst in instances:
        inst_type = inst["type"]
        type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
    
    print(f"\nInstance types:")
    for inst_type, count in type_counts.items():
        print(f"  {inst_type}: {count}")
    
    # Get unique states
    states = {inst["state"] for inst in instances}
    print(f"\nUnique states: {states}")
    
    # Extract just IDs
    instance_ids = [inst["id"] for inst in instances]
    print(f"\nAll IDs: {instance_ids}")
    
    print("\n💡 List comprehensions make filtering/transforming easy!")


# ============================================================================
# 7. Real-World: Configuration Management
# ============================================================================

def demo_config_management():
    """
    Real-world: Manage application configuration.
    
    Common patterns:
    - Default configs
    - Environment overrides
    - Merging configs
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Configuration Management")
    print("=" * 70)
    
    # Default configuration
    default_config = {
        "host": "localhost",
        "port": 8080,
        "debug": False,
        "workers": 4,
        "timeout": 30,
        "features": {"cache": True, "metrics": False}
    }
    
    # Environment-specific overrides
    prod_overrides = {
        "host": "0.0.0.0",
        "debug": False,
        "workers": 16,
        "features": {"metrics": True}
    }
    
    # Merge configs (prod overrides default)
    config = default_config.copy()
    config.update(prod_overrides)
    
    # Deep merge for nested dicts
    if "features" in prod_overrides:
        config["features"].update(prod_overrides["features"])
    
    print("Production config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate required fields
    required_fields = ["host", "port", "workers"]
    missing = [field for field in required_fields if field not in config]
    
    if missing:
        print(f"\n❌ Missing required fields: {missing}")
    else:
        print(f"\n✅ All required fields present")
    
    # Extract feature flags
    enabled_features = [
        feature for feature, enabled in config["features"].items()
        if enabled
    ]
    print(f"\nEnabled features: {enabled_features}")
    
    print("\n💡 Dicts are perfect for configuration management!")


# ============================================================================
# 8. Real-World: Data Deduplication
# ============================================================================

def demo_deduplication():
    """
    Real-world: Remove duplicates from data.
    
    Common scenarios:
    - User IDs from multiple sources
    - Email addresses
    - Log deduplication
    """
    print("\n" + "=" * 70)
    print("8. Real-World: Data Deduplication")
    print("=" * 70)
    
    # Multiple data sources with duplicates
    database_users = ["alice@ex.com", "bob@ex.com", "charlie@ex.com"]
    api_users = ["bob@ex.com", "david@ex.com", "alice@ex.com"]
    csv_users = ["charlie@ex.com", "eve@ex.com", "alice@ex.com"]
    
    print("Data sources:")
    print(f"  Database: {database_users}")
    print(f"  API: {api_users}")
    print(f"  CSV: {csv_users}")
    
    # Combine all (with duplicates)
    all_users = database_users + api_users + csv_users
    print(f"\nCombined ({len(all_users)} total): {all_users}")
    
    # Remove duplicates with set
    unique_users = set(all_users)
    print(f"\nUnique users ({len(unique_users)}): {sorted(unique_users)}")
    
    # Find duplicates
    seen = set()
    duplicates = []
    for user in all_users:
        if user in seen:
            if user not in duplicates:
                duplicates.append(user)
        else:
            seen.add(user)
    
    print(f"\nDuplicates found: {duplicates}")
    
    # Count occurrences
    counts = {}
    for user in all_users:
        counts[user] = counts.get(user, 0) + 1
    
    print(f"\nOccurrence counts:")
    for user, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {user}: {count}x")
    
    print("\n💡 Sets are the fastest way to remove duplicates!")


# ============================================================================
# 9. List Comprehensions - Pythonic Filtering/Mapping
# ============================================================================

def demo_list_comprehensions():
    """
    List comprehensions are a Pythonic way to create lists.
    More concise than for loops, similar to JS map/filter.
    """
    print("\n" + "=" * 70)
    print("9. List Comprehensions - Pythonic Way")
    print("=" * 70)
    
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Traditional way (verbose)
    evens_old = []
    for n in numbers:
        if n % 2 == 0:
            evens_old.append(n)
    
    # Comprehension (Pythonic!)
    evens = [n for n in numbers if n % 2 == 0]
    
    print(f"Even numbers: {evens}")
    
    # Map operation
    squared = [n ** 2 for n in numbers]
    print(f"Squared: {squared}")
    
    # Filter and map combined
    even_squares = [n ** 2 for n in numbers if n % 2 == 0]
    print(f"Even squares: {even_squares}")
    
    # Dict comprehension
    squares_dict = {n: n ** 2 for n in numbers}
    print(f"\nDict comprehension: {squares_dict}")
    
    # Set comprehension
    unique_lengths = {len(word) for word in ["hi", "hello", "hey", "hi"]}
    print(f"Set comprehension: {unique_lengths}")
    
    # Real-world: Extract fields from list of dicts
    users = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]
    
    names = [user["name"] for user in users]
    adults = [user for user in users if user["age"] >= 30]
    
    print(f"\nNames: {names}")
    print(f"Adults: {adults}")
    
    print("\n💡 List comprehensions are more Pythonic than for loops!")


# ============================================================================
# 10. Common Patterns and Idioms
# ============================================================================

def demo_common_patterns():
    """
    Common Python patterns you'll use frequently.
    """
    print("\n" + "=" * 70)
    print("10. Common Patterns")
    print("=" * 70)
    
    # Pattern 1: Checking if collection is empty
    items = []
    if not items:  # Pythonic!
        print("✅ Empty list check: if not items")
    
    # Pattern 2: Default dict values
    config = {"host": "localhost"}
    port = config.get("port", 8080)  # Returns 8080 if "port" missing
    print(f"✅ Default value: port = config.get('port', 8080) → {port}")
    
    # Pattern 3: Swapping variables
    a, b = 10, 20
    a, b = b, a  # Swap!
    print(f"✅ Swap: a={a}, b={b}")
    
    # Pattern 4: Enumerate with index
    print("\n✅ Enumerate for index + value:")
    for i, value in enumerate(["a", "b", "c"], start=1):
        print(f"  {i}. {value}")
    
    # Pattern 5: Zip two lists
    names = ["Alice", "Bob", "Charlie"]
    ages = [30, 25, 35]
    for name, age in zip(names, ages):
        print(f"  {name}: {age}")
    
    # Pattern 6: Unpacking with *
    first, *middle, last = [1, 2, 3, 4, 5]
    print(f"\n✅ Unpacking: first={first}, middle={middle}, last={last}")
    
    # Pattern 7: Combining dicts (3.9+)
    defaults = {"a": 1, "b": 2}
    overrides = {"b": 3, "c": 4}
    merged = defaults | overrides
    print(f"✅ Merge dicts: {merged}")
    
    print("\n💡 Learn these patterns - you'll use them constantly!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Collections for Go/JS Engineers\n")
    
    demo_lists()
    demo_dictionaries()
    demo_sets()
    demo_tuples()
    demo_log_processing()
    demo_api_responses()
    demo_config_management()
    demo_deduplication()
    demo_list_comprehensions()
    demo_common_patterns()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Lists: Ordered, mutable, use for sequences
2. Dicts: Key-value pairs, use for mappings/objects
3. Sets: Unique items, fast membership testing
4. Tuples: Immutable, use for fixed data
5. List comprehensions: Pythonic filtering/mapping
6. Use 'in' for membership testing
7. Use .get() for safe dict access
8. Negative indexing: [-1] for last item
9. Slicing: [start:end:step]
10. Unpacking: a, b = (1, 2)

When to use:
- Lists: Logs, API responses, ordered data
- Dicts: Configs, counters, objects
- Sets: Deduplication, membership testing
- Tuples: Multiple returns, dict keys
""")
    
    print("🎯 Next: 03_control_flow.py - if/for/while, comprehensions")


if __name__ == "__main__":
    main()
