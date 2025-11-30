"""
Files and I/O - Python Essentials for Go/JS Engineers

Learn Python's file operations and data formats (JSON, CSV).
Context managers make file handling cleaner than Go/JS.

Run: poetry run python 00-python-essentials/examples/06_files_and_io.py
"""

import json
import csv
from pathlib import Path
from typing import Any


# ============================================================================
# 1. Reading Files - Context Managers
# ============================================================================

def demo_reading_files():
    """
    Python's 'with' statement handles file closing automatically.
    Much cleaner than manual close() calls!
    """
    print("=" * 70)
    print("1. Reading Files with Context Managers")
    print("=" * 70)
    
    # Create a test file
    test_file = Path("demo.txt")
    test_file.write_text("Hello, Python!\nThis is a test file.\nLine 3.")
    
    # Read entire file
    with open(test_file, "r") as f:
        content = f.read()
    print("Full content:")
    print(content)
    
    # Read lines as list
    with open(test_file, "r") as f:
        lines = f.readlines()
    print("\nLines as list:")
    for i, line in enumerate(lines, 1):
        print(f"  {i}: {line.strip()}")
    
    # Read line by line (memory efficient)
    print("\nIterating lines:")
    with open(test_file, "r") as f:
        for i, line in enumerate(f, 1):
            print(f"  {i}: {line.strip()}")
    
    # Cleanup
    test_file.unlink()
    
    print("\n💡 'with' automatically closes files - safer than manual close()!")


# ============================================================================
# 2. Writing Files
# ============================================================================

def demo_writing_files():
    """
    Writing files with different modes.
    """
    print("\n" + "=" * 70)
    print("2. Writing Files")
    print("=" * 70)
    
    # Write mode (overwrites existing file)
    test_file = Path("output.txt")
    
    with open(test_file, "w") as f:
        f.write("Line 1\n")
        f.write("Line 2\n")
    print("✅ Wrote 2 lines (write mode)")
    
    # Append mode (adds to existing file)
    with open(test_file, "a") as f:
        f.write("Line 3\n")
        f.write("Line 4\n")
    print("✅ Appended 2 lines (append mode)")
    
    # Read back
    content = test_file.read_text()
    print("\nFile content:")
    print(content)
    
    # Write multiple lines at once
    lines = ["First line", "Second line", "Third line"]
    with open(test_file, "w") as f:
        f.write("\n".join(lines))
    print("✅ Wrote lines with join()")
    
    # Cleanup
    test_file.unlink()
    
    print("\n💡 Use 'w' to overwrite, 'a' to append!")


# ============================================================================
# 3. Working with JSON - Most Common Format
# ============================================================================

def demo_json_operations():
    """
    JSON is the most common data format for APIs and configs.
    """
    print("\n" + "=" * 70)
    print("3. Working with JSON")
    print("=" * 70)
    
    # Python dict to JSON
    config = {
        "host": "localhost",
        "port": 8080,
        "debug": True,
        "features": ["cache", "metrics"],
        "limits": {
            "max_connections": 100,
            "timeout": 30
        }
    }
    
    # Write JSON to file
    json_file = Path("config.json")
    with open(json_file, "w") as f:
        json.dump(config, f, indent=2)
    print("✅ Wrote JSON to file")
    
    # Read JSON from file
    with open(json_file, "r") as f:
        loaded_config = json.load(f)
    print(f"✅ Loaded JSON: {loaded_config}")
    
    # Convert dict to JSON string
    json_str = json.dumps(config, indent=2)
    print("\nJSON string:")
    print(json_str)
    
    # Parse JSON string
    parsed = json.loads(json_str)
    print(f"\n✅ Parsed JSON: {parsed['host']}:{parsed['port']}")
    
    # Handle JSON errors
    try:
        invalid_json = '{"key": invalid}'
        data = json.loads(invalid_json)
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")
    
    # Cleanup
    json_file.unlink()
    
    print("\n💡 JSON is perfect for configs and API data!")


# ============================================================================
# 4. Working with CSV - Tabular Data
# ============================================================================

def demo_csv_operations():
    """
    CSV for spreadsheet-like data.
    """
    print("\n" + "=" * 70)
    print("4. Working with CSV")
    print("=" * 70)
    
    csv_file = Path("users.csv")
    
    # Write CSV
    users = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "SF"},
        {"name": "Charlie", "age": 35, "city": "LA"},
    ]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
        writer.writeheader()
        writer.writerows(users)
    print("✅ Wrote CSV file")
    
    # Read CSV
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        print("\nRead CSV:")
        for row in reader:
            print(f"  {row['name']}: {row['age']} years old, {row['city']}")
    
    # Write CSV with list of lists
    data = [
        ["Product", "Price", "Stock"],
        ["Laptop", "999.99", "15"],
        ["Mouse", "29.99", "100"],
        ["Keyboard", "79.99", "50"],
    ]
    
    csv_file2 = Path("products.csv")
    with open(csv_file2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print("\n✅ Wrote CSV with writer")
    
    # Read with csv.reader
    with open(csv_file2, "r") as f:
        reader = csv.reader(f)
        print("\nRead CSV:")
        for row in reader:
            print(f"  {row}")
    
    # Cleanup
    csv_file.unlink()
    csv_file2.unlink()
    
    print("\n💡 CSV for spreadsheets, JSON for structured data!")


# ============================================================================
# 5. Path Operations with pathlib
# ============================================================================

def demo_path_operations():
    """
    pathlib is the modern way to handle file paths.
    Much better than os.path!
    """
    print("\n" + "=" * 70)
    print("5. Path Operations with pathlib")
    print("=" * 70)
    
    # Create paths
    path = Path("data") / "logs" / "app.log"
    print(f"Path: {path}")
    print(f"Parent: {path.parent}")
    print(f"Name: {path.name}")
    print(f"Suffix: {path.suffix}")
    print(f"Stem: {path.stem}")
    
    # Check existence
    config_path = Path("config.json")
    print(f"\nconfig.json exists? {config_path.exists()}")
    
    # Create directories
    log_dir = Path("temp_logs")
    log_dir.mkdir(exist_ok=True)  # Won't error if exists
    print(f"✅ Created directory: {log_dir}")
    
    # Create nested directories
    nested = Path("temp_data/logs/2024")
    nested.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created nested: {nested}")
    
    # List directory contents
    print(f"\nContents of current directory:")
    for item in Path(".").iterdir():
        if item.is_file():
            print(f"  📄 {item.name}")
        elif item.is_dir():
            print(f"  📁 {item.name}/")
    
    # Find files by pattern
    print(f"\nPython files (*.py):")
    for py_file in Path(".").glob("*.py"):
        print(f"  {py_file.name}")
    
    # Cleanup
    log_dir.rmdir()
    for item in nested.parents:
        if item.exists() and item != Path("."):
            item.rmdir()
    
    print("\n💡 pathlib is cleaner than os.path!")


# ============================================================================
# 6. Real-World: Configuration Files
# ============================================================================

def demo_config_files():
    """
    Real-world: Load and merge configuration files.
    
    Common pattern:
    - Default config in code
    - Override with config file
    - Override with env vars
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Configuration Management")
    print("=" * 70)
    
    # Default configuration
    default_config = {
        "host": "localhost",
        "port": 8080,
        "debug": False,
        "workers": 4,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp"
        }
    }
    
    # Save default config
    config_file = Path("app_config.json")
    with open(config_file, "w") as f:
        json.dump(default_config, f, indent=2)
    print("✅ Created default config")
    
    # Load and merge with overrides
    def load_config(config_path: Path) -> dict:
        """Load config with defaults."""
        config = default_config.copy()
        
        if config_path.exists():
            with open(config_path, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
            print(f"✅ Loaded config from {config_path}")
        else:
            print(f"⚠️  Config not found, using defaults")
        
        return config
    
    # Test loading
    config = load_config(config_file)
    print(f"\nLoaded config:")
    print(f"  Host: {config['host']}")
    print(f"  Port: {config['port']}")
    print(f"  Debug: {config['debug']}")
    print(f"  Workers: {config['workers']}")
    
    # Validate required fields
    required_fields = ["host", "port", "database"]
    missing = [field for field in required_fields if field not in config]
    
    if missing:
        print(f"\n❌ Missing required fields: {missing}")
    else:
        print(f"\n✅ All required fields present")
    
    # Cleanup
    config_file.unlink()
    
    print("\n💡 JSON config files are easy to read and edit!")


# ============================================================================
# 7. Real-World: Log File Processing
# ============================================================================

def demo_log_processing():
    """
    Real-world: Parse and analyze log files.
    
    Common tasks:
    - Find errors
    - Extract timestamps
    - Count occurrences
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Log File Processing")
    print("=" * 70)
    
    # Create sample log file
    log_entries = [
        "2024-01-15 10:00:00 INFO Server started",
        "2024-01-15 10:00:01 INFO User alice logged in",
        "2024-01-15 10:00:02 ERROR Database connection failed",
        "2024-01-15 10:00:03 INFO Processing request",
        "2024-01-15 10:00:04 ERROR API timeout",
        "2024-01-15 10:00:05 WARN High memory usage: 85%",
        "2024-01-15 10:00:06 ERROR Database connection failed",
        "2024-01-15 10:00:07 INFO Request completed",
    ]
    
    log_file = Path("app.log")
    log_file.write_text("\n".join(log_entries))
    print(f"✅ Created log file with {len(log_entries)} entries")
    
    # Process log file
    errors = []
    warnings = []
    error_counts = {}
    
    with open(log_file, "r") as f:
        for line in f:
            if "ERROR" in line:
                errors.append(line.strip())
                # Count error types
                error_msg = line.split("ERROR ", 1)[1].strip()
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
            elif "WARN" in line:
                warnings.append(line.strip())
    
    print(f"\n📊 Log Analysis:")
    print(f"  Total errors: {len(errors)}")
    print(f"  Total warnings: {len(warnings)}")
    
    print(f"\nError breakdown:")
    for error, count in error_counts.items():
        print(f"  {error}: {count}x")
    
    print(f"\nError details:")
    for error in errors:
        print(f"  {error}")
    
    # Extract errors to separate file
    error_file = Path("errors.log")
    with open(error_file, "w") as f:
        f.write("\n".join(errors))
    print(f"\n✅ Extracted errors to {error_file}")
    
    # Cleanup
    log_file.unlink()
    error_file.unlink()
    
    print("\n💡 Log processing is common in DevOps and monitoring!")


# ============================================================================
# 8. Real-World: Data Export/Import
# ============================================================================

def demo_data_export_import():
    """
    Real-world: Export data to CSV/JSON for analysis.
    
    Common scenarios:
    - Export database results
    - Import bulk data
    - Data migration
    """
    print("\n" + "=" * 70)
    print("8. Real-World: Data Export/Import")
    print("=" * 70)
    
    # Simulate database query results
    users = [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": True},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": True},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": False},
    ]
    
    # Export to JSON
    json_export = Path("users_export.json")
    with open(json_export, "w") as f:
        json.dump(users, f, indent=2)
    print(f"✅ Exported {len(users)} users to JSON")
    
    # Export to CSV
    csv_export = Path("users_export.csv")
    with open(csv_export, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "email", "active"])
        writer.writeheader()
        writer.writerows(users)
    print(f"✅ Exported {len(users)} users to CSV")
    
    # Import from JSON
    with open(json_export, "r") as f:
        imported_json = json.load(f)
    print(f"\n✅ Imported {len(imported_json)} users from JSON")
    
    # Import from CSV
    imported_csv = []
    with open(csv_export, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types (CSV reads everything as strings)
            imported_csv.append({
                "id": int(row["id"]),
                "name": row["name"],
                "email": row["email"],
                "active": row["active"] == "True"
            })
    print(f"✅ Imported {len(imported_csv)} users from CSV")
    
    # Validate imported data
    for user in imported_csv:
        print(f"  {user['name']}: active={user['active']}")
    
    # Cleanup
    json_export.unlink()
    csv_export.unlink()
    
    print("\n💡 Use JSON for nested data, CSV for tabular data!")


# ============================================================================
# 9. File Operations Best Practices
# ============================================================================

def demo_best_practices():
    """
    Best practices for file operations.
    """
    print("\n" + "=" * 70)
    print("9. File Operations Best Practices")
    print("=" * 70)
    
    print("✅ DO:")
    print("  1. Always use 'with' for file operations")
    print("  2. Use pathlib.Path instead of strings")
    print("  3. Use Path.read_text() for small files")
    print("  4. Handle FileNotFoundError explicitly")
    print("  5. Use json.load/dump for JSON")
    print("  6. Use csv.DictReader/DictWriter for CSV")
    print("  7. Specify encoding (utf-8) for text files")
    print("  8. Use exist_ok=True for mkdir")
    
    print("\n❌ DON'T:")
    print("  1. Forget to close files (use 'with')")
    print("  2. Load huge files into memory at once")
    print("  3. Use string concatenation for paths")
    print("  4. Ignore file operation errors")
    print("  5. Hardcode file paths (use Path)")
    
    # Good example
    print("\n✅ GOOD:")
    print('''
    from pathlib import Path
    
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    ''')
    
    # Bad example
    print("❌ BAD:")
    print('''
    # No 'with', manual close
    f = open("config.json")
    config = json.load(f)
    f.close()  # What if exception before this?
    ''')
    
    print("\n💡 Use 'with' and pathlib for safe file operations!")


# ============================================================================
# 10. Error Handling with Files
# ============================================================================

def demo_file_error_handling():
    """
    Handle file operation errors gracefully.
    """
    print("\n" + "=" * 70)
    print("10. File Error Handling")
    print("=" * 70)
    
    def safe_read_json(path: Path) -> dict | None:
        """Safely read JSON file with error handling."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {path}: {e}")
            return None
        except PermissionError:
            print(f"❌ Permission denied: {path}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    # Test with various scenarios
    test_cases = [
        Path("nonexistent.json"),
        Path("invalid.json"),
    ]
    
    # Create invalid JSON file
    Path("invalid.json").write_text("{ invalid json }")
    
    for path in test_cases:
        print(f"\nTesting: {path}")
        result = safe_read_json(path)
        if result:
            print(f"  ✅ Loaded: {result}")
        else:
            print(f"  ⚠️  Could not load")
    
    # Cleanup
    Path("invalid.json").unlink()
    
    print("\n💡 Always handle file errors - files are unreliable!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Files and I/O for Go/JS Engineers\n")
    
    demo_reading_files()
    demo_writing_files()
    demo_json_operations()
    demo_csv_operations()
    demo_path_operations()
    demo_config_files()
    demo_log_processing()
    demo_data_export_import()
    demo_best_practices()
    demo_file_error_handling()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Use 'with open()' for automatic file closing
2. Use pathlib.Path for path operations
3. JSON: json.load/dump for files, loads/dumps for strings
4. CSV: csv.DictReader/DictWriter for dicts
5. Path.read_text() / write_text() for simple operations
6. Handle FileNotFoundError explicitly
7. Use Path.glob() to find files by pattern
8. mkdir(parents=True, exist_ok=True) for directories
9. Always handle encoding (default is utf-8)
10. Process large files line by line

File modes:
- 'r': Read (default)
- 'w': Write (overwrite)
- 'a': Append
- 'rb'/'wb': Binary mode

Common operations:
- Read: Path.read_text() or open().read()
- Write: Path.write_text() or open().write()
- JSON: json.load/dump
- CSV: csv.DictReader/DictWriter
- List files: Path.glob('*.py')
""")
    
    print("🎯 Next: 07_strings.py - String operations, formatting")


if __name__ == "__main__":
    main()
