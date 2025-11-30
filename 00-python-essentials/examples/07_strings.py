"""
Strings - Python Essentials for Go/JS Engineers

Learn Python's powerful string operations and formatting.
Python strings are more feature-rich than Go, similar to JS.

Run: poetry run python 00-python-essentials/examples/07_strings.py
"""

import re
from typing import List


# ============================================================================
# 1. String Basics - Similar to Go/JS
# ============================================================================

def demo_string_basics():
    """
    Strings in Python are immutable sequences of Unicode characters.
    """
    print("=" * 70)
    print("1. String Basics")
    print("=" * 70)
    
    # Creating strings
    single = 'Hello'
    double = "World"
    multiline = """This is a
    multi-line string"""
    raw = r"C:\Users\name\file.txt"  # Raw string (no escaping)
    
    print(f"Single quotes: {single}")
    print(f"Double quotes: {double}")
    print(f"Multiline: {multiline}")
    print(f"Raw string: {raw}")
    
    # String concatenation
    greeting = "Hello" + " " + "World"
    print(f"\nConcatenation: {greeting}")
    
    # String repetition
    separator = "=" * 20
    print(f"Repetition: {separator}")
    
    # Length
    text = "Hello, World!"
    print(f"\nLength of '{text}': {len(text)}")
    
    # Indexing and slicing
    print(f"First char: {text[0]}")
    print(f"Last char: {text[-1]}")
    print(f"First 5: {text[:5]}")
    print(f"Last 6: {text[-6:]}")
    print(f"Every 2nd: {text[::2]}")
    print(f"Reversed: {text[::-1]}")
    
    print("\n💡 Strings are immutable - methods return new strings!")


# ============================================================================
# 2. String Formatting - Multiple Approaches
# ============================================================================

def demo_string_formatting():
    """
    Python has several ways to format strings.
    f-strings (Python 3.6+) are the modern, preferred way.
    """
    print("\n" + "=" * 70)
    print("2. String Formatting")
    print("=" * 70)
    
    name = "Alice"
    age = 30
    price = 19.99
    
    # f-strings (modern, recommended!)
    print("f-strings (best):")
    print(f"  Hello, {name}!")
    print(f"  {name} is {age} years old")
    print(f"  Price: ${price:.2f}")
    print(f"  Math: {10 + 5} = 15")
    
    # .format() (older, still common)
    print("\n.format():")
    print("  Hello, {}!".format(name))
    print("  {} is {} years old".format(name, age))
    print("  Price: ${:.2f}".format(price))
    
    # % formatting (old style, avoid)
    print("\n% formatting (avoid):")
    print("  Hello, %s!" % name)
    print("  %s is %d years old" % (name, age))
    
    # f-string expressions
    print("\nf-string expressions:")
    print(f"  Uppercase: {name.upper()}")
    print(f"  Conditional: {'Adult' if age >= 18 else 'Minor'}")
    print(f"  List: {[x**2 for x in range(5)]}")
    
    # Number formatting
    big_number = 1234567.89
    print("\nNumber formatting:")
    print(f"  Comma separator: {big_number:,}")
    print(f"  2 decimals: {big_number:.2f}")
    print(f"  Scientific: {big_number:.2e}")
    print(f"  Percentage: {0.856:.1%}")
    
    # Alignment
    print("\nAlignment:")
    print(f"  Left: |{name:<15}|")
    print(f"  Right: |{name:>15}|")
    print(f"  Center: |{name:^15}|")
    
    print("\n💡 Use f-strings - they're fast, readable, and powerful!")


# ============================================================================
# 3. Common String Methods
# ============================================================================

def demo_string_methods():
    """
    Python strings have many useful methods.
    """
    print("\n" + "=" * 70)
    print("3. Common String Methods")
    print("=" * 70)
    
    text = "  Hello, World!  "
    
    # Case conversion
    print("Case conversion:")
    print(f"  Upper: {text.upper()}")
    print(f"  Lower: {text.lower()}")
    print(f"  Title: {text.title()}")
    print(f"  Capitalize: {text.capitalize()}")
    print(f"  Swap: {text.swapcase()}")
    
    # Whitespace
    print("\nWhitespace:")
    print(f"  Original: |{text}|")
    print(f"  Strip: |{text.strip()}|")
    print(f"  Lstrip: |{text.lstrip()}|")
    print(f"  Rstrip: |{text.rstrip()}|")
    
    # Searching
    print("\nSearching:")
    print(f"  'World' in text: {'World' in text}")
    print(f"  Starts with 'Hello': {text.strip().startswith('Hello')}")
    print(f"  Ends with '!': {text.strip().endswith('!')}")
    print(f"  Find 'World': {text.find('World')}")
    print(f"  Count 'l': {text.count('l')}")
    
    # Replacing
    print("\nReplacing:")
    print(f"  Replace 'World' with 'Python': {text.replace('World', 'Python')}")
    print(f"  Replace 'l' with 'L': {text.replace('l', 'L')}")
    
    # Splitting
    print("\nSplitting:")
    words = "apple,banana,orange"
    print(f"  Split by comma: {words.split(',')}")
    sentence = "Hello World Python"
    print(f"  Split by space: {sentence.split()}")
    lines = "line1\nline2\nline3"
    print(f"  Split by newline: {lines.splitlines()}")
    
    # Joining
    print("\nJoining:")
    items = ["apple", "banana", "orange"]
    print(f"  Join with comma: {', '.join(items)}")
    print(f"  Join with newline: \\n{chr(10).join(items)}")
    
    print("\n💡 String methods are very powerful - use them!")


# ============================================================================
# 4. String Validation
# ============================================================================

def demo_string_validation():
    """
    Check string properties - useful for validation.
    """
    print("\n" + "=" * 70)
    print("4. String Validation")
    print("=" * 70)
    
    # Type checks
    test_strings = {
        "abc123": "alphanumeric",
        "abc": "alphabetic",
        "123": "numeric",
        "   ": "whitespace",
        "Hello World": "mixed",
    }
    
    for string, desc in test_strings.items():
        print(f"\n'{string}' ({desc}):")
        print(f"  isalnum(): {string.isalnum()}")
        print(f"  isalpha(): {string.isalpha()}")
        print(f"  isdigit(): {string.isdigit()}")
        print(f"  isspace(): {string.isspace()}")
        print(f"  isupper(): {string.isupper()}")
        print(f"  islower(): {string.islower()}")
    
    print("\n💡 Use is* methods for input validation!")


# ============================================================================
# 5. Regular Expressions - Pattern Matching
# ============================================================================

def demo_regex():
    """
    Regular expressions for pattern matching.
    Similar to JS regex, more powerful than Go's.
    """
    print("\n" + "=" * 70)
    print("5. Regular Expressions")
    print("=" * 70)
    
    # Find pattern
    text = "Contact: alice@example.com or bob@test.com"
    emails = re.findall(r'\S+@\S+', text)
    print(f"Find emails: {emails}")
    
    # Match pattern
    phone = "555-123-4567"
    pattern = r'^\d{3}-\d{3}-\d{4}$'
    if re.match(pattern, phone):
        print(f"✅ Valid phone: {phone}")
    
    # Replace with regex
    text = "Price: $19.99 and $29.99"
    updated = re.sub(r'\$(\d+\.\d{2})', r'USD \1', text)
    print(f"Replace: {updated}")
    
    # Split by pattern
    text = "one,two;three:four"
    parts = re.split(r'[,;:]', text)
    print(f"Split by delimiters: {parts}")
    
    # Extract groups
    log = "2024-01-15 10:00:00 ERROR Database connection failed"
    match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+) (.+)', log)
    if match:
        date, time, level, message = match.groups()
        print(f"\nExtracted:")
        print(f"  Date: {date}")
        print(f"  Time: {time}")
        print(f"  Level: {level}")
        print(f"  Message: {message}")
    
    print("\n💡 Regex is powerful for text processing!")


# ============================================================================
# 6. Real-World: Email Validation
# ============================================================================

def demo_email_validation():
    """
    Real-world: Validate email addresses.
    
    Common in:
    - User registration
    - Form validation
    - Data cleaning
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Email Validation")
    print("=" * 70)
    
    def is_valid_email(email: str) -> bool:
        """Simple email validation."""
        # Basic pattern (not RFC-compliant, but good enough)
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    test_emails = [
        ("alice@example.com", True),
        ("bob.smith@company.co.uk", True),
        ("invalid.email", False),
        ("@example.com", False),
        ("user@", False),
        ("user name@example.com", False),
    ]
    
    print("Email validation:")
    for email, expected in test_emails:
        valid = is_valid_email(email)
        status = "✅" if valid == expected else "❌"
        print(f"  {status} '{email}': {valid}")
    
    print("\n💡 Regex is perfect for pattern validation!")


# ============================================================================
# 7. Real-World: Log Parsing
# ============================================================================

def demo_log_parsing():
    """
    Real-world: Parse structured log entries.
    
    Common in DevOps:
    - Analyzing application logs
    - Extracting metrics
    - Finding patterns
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Log Parsing")
    print("=" * 70)
    
    logs = [
        "2024-01-15 10:00:00 ERROR Database connection timeout (30s)",
        "2024-01-15 10:00:01 INFO Request processed in 45ms",
        "2024-01-15 10:00:02 ERROR API call failed after 3 retries",
        "2024-01-15 10:00:03 WARN Memory usage: 85%",
    ]
    
    # Parse log entries
    pattern = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+) (.+)'
    
    print("Parsed logs:")
    for log in logs:
        match = re.match(pattern, log)
        if match:
            date, time, level, message = match.groups()
            print(f"  [{level}] {date} {time}")
            print(f"    {message}")
    
    # Extract timing information
    print("\nExtract timings:")
    for log in logs:
        # Find numbers followed by time units
        timings = re.findall(r'(\d+)(ms|s)', log)
        if timings:
            for value, unit in timings:
                print(f"  {log.split()[-1]}: {value}{unit}")
    
    # Count by log level
    print("\nCount by level:")
    levels = {}
    for log in logs:
        match = re.search(r'\s(\w+)\s', log)
        if match:
            level = match.group(1)
            levels[level] = levels.get(level, 0) + 1
    
    for level, count in levels.items():
        print(f"  {level}: {count}")
    
    print("\n💡 String parsing + regex = powerful log analysis!")


# ============================================================================
# 8. Real-World: URL Parsing
# ============================================================================

def demo_url_parsing():
    """
    Real-world: Parse and manipulate URLs.
    """
    print("\n" + "=" * 70)
    print("8. Real-World: URL Parsing")
    print("=" * 70)
    
    url = "https://api.example.com/v1/users?page=2&limit=50"
    
    # Extract components
    pattern = r'(https?)://([^/]+)(/[^?]*)?\??(.*)' 
    match = re.match(pattern, url)
    
    if match:
        protocol, domain, path, query = match.groups()
        print(f"URL: {url}")
        print(f"  Protocol: {protocol}")
        print(f"  Domain: {domain}")
        print(f"  Path: {path or '/'}")
        print(f"  Query: {query}")
    
    # Parse query parameters
    if query:
        print("\nQuery parameters:")
        params = {}
        for pair in query.split('&'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[key] = value
                print(f"  {key}: {value}")
    
    # Build URL from parts
    def build_url(base: str, endpoint: str, params: dict) -> str:
        """Build URL with query parameters."""
        url = f"{base}{endpoint}"
        if params:
            query = '&'.join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
        return url
    
    new_url = build_url(
        "https://api.example.com",
        "/v1/search",
        {"q": "python", "page": "1", "per_page": "20"}
    )
    print(f"\nBuilt URL: {new_url}")
    
    print("\n💡 String manipulation is essential for API work!")


# ============================================================================
# 9. Real-World: Text Cleaning
# ============================================================================

def demo_text_cleaning():
    """
    Real-world: Clean and normalize text data.
    
    Common in:
    - Data preprocessing
    - NLP pipelines
    - Search indexing
    """
    print("\n" + "=" * 70)
    print("9. Real-World: Text Cleaning")
    print("=" * 70)
    
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    dirty_texts = [
        "  Hello,   World!  ",
        "Python@3.10   is   great!!!",
        "Remove\n\nnewlines\tand\ttabs",
        "   Lots    of    spaces   ",
    ]
    
    print("Text cleaning:")
    for text in dirty_texts:
        cleaned = clean_text(text)
        print(f"  Before: '{text}'")
        print(f"  After:  '{cleaned}'")
        print()
    
    print("💡 Text cleaning is crucial for data quality!")


# ============================================================================
# 10. String Best Practices
# ============================================================================

def demo_best_practices():
    """
    Best practices for working with strings.
    """
    print("\n" + "=" * 70)
    print("10. String Best Practices")
    print("=" * 70)
    
    print("✅ DO:")
    print("  1. Use f-strings for formatting")
    print("  2. Use .join() for concatenating many strings")
    print("  3. Use raw strings (r'') for regex patterns")
    print("  4. Use triple quotes for multiline strings")
    print("  5. Use .strip() before validation")
    print("  6. Use 'in' for substring checks")
    print("  7. Use str.format() for templates")
    print("  8. Compile regex if using repeatedly")
    
    print("\n❌ DON'T:")
    print("  1. Use + in loops for concatenation")
    print("  2. Forget strings are immutable")
    print("  3. Use == for case-insensitive comparison")
    print("  4. Build complex regex without testing")
    
    # Good: Use join for concatenation
    print("\n✅ GOOD (join):")
    words = ["Python", "is", "awesome"]
    sentence = " ".join(words)
    print(f"  {sentence}")
    
    # Bad: Use + in loop
    print("\n❌ BAD (+ in loop):")
    print("  sentence = ''")
    print("  for word in words:")
    print("      sentence += word + ' '  # Creates new string each time!")
    
    # Case-insensitive comparison
    print("\n✅ GOOD (case-insensitive):")
    text1 = "Hello"
    text2 = "hello"
    print(f"  {text1.lower() == text2.lower()}")
    
    # Compile regex for reuse
    print("\n✅ GOOD (compile regex):")
    print("  email_pattern = re.compile(r'\\S+@\\S+')")
    print("  if email_pattern.match(email):")
    print("      ...")
    
    print("\n💡 Follow best practices for maintainable code!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐍 Python Strings for Go/JS Engineers\n")
    
    demo_string_basics()
    demo_string_formatting()
    demo_string_methods()
    demo_string_validation()
    demo_regex()
    demo_email_validation()
    demo_log_parsing()
    demo_url_parsing()
    demo_text_cleaning()
    demo_best_practices()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. f-strings: f"Hello {name}" - modern, fast, readable
2. String slicing: text[start:end:step]
3. Immutable: methods return new strings
4. .strip(), .split(), .join() are essential
5. Use 'in' for substring checks
6. Regex: re.match(), re.search(), re.findall()
7. .lower() for case-insensitive comparison
8. Use .join() for concatenating many strings
9. Raw strings: r"C:\\path" for regex/paths
10. Triple quotes for multiline strings

Common methods:
- .upper(), .lower(), .title()
- .strip(), .lstrip(), .rstrip()
- .split(), .splitlines()
- .join()
- .replace()
- .startswith(), .endswith()
- .find(), .count()
- .isalpha(), .isdigit(), .isalnum()

Formatting:
- f"{value}"           - basic
- f"{value:.2f}"       - 2 decimals
- f"{value:,}"         - comma separator
- f"{value:>10}"       - right align
- f"{value:.1%}"       - percentage
""")
    
    print("🎯 Next: 08_http_and_apis.py - Making API calls")


if __name__ == "__main__":
    main()
