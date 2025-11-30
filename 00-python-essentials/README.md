# Module 0: Python Essentials

**For Senior Engineers coming from Go/JavaScript/DevOps**

This module covers fundamental Python concepts with real-world examples you can relate to. We'll compare Python to Go and JavaScript where relevant.

## 🎯 Learning Objectives

- Master Python's basic syntax and idioms
- Understand Python's dynamic typing vs Go's static typing
- Learn Pythonic ways to handle common tasks
- See how Python differs from JavaScript's approach
- Build confidence with hands-on, practical examples

## 📚 Topics Covered

### 1. **Variables and Types** (`01_variables_and_types.py`)
- Dynamic typing vs Go's static typing
- Type inference and duck typing
- Common types: str, int, float, bool, None
- Type conversion and gotchas
- Real-world: Parsing API responses, config files

### 2. **Collections** (`02_collections.py`)
- Lists (like JS arrays, Go slices)
- Dictionaries (like JS objects, Go maps)
- Sets and Tuples
- When to use each collection type
- Real-world: Processing logs, API responses, batch data

### 3. **Control Flow** (`03_control_flow.py`)
- if/elif/else (cleaner than Go's if/else)
- for loops (simpler than Go, like JS for...of)
- while loops
- List comprehensions (Pythonic!)
- Real-world: Data filtering, validation, transformation

### 4. **Functions** (`04_functions.py`)
- Function basics (vs Go functions)
- Arguments: positional, keyword, default, *args, **kwargs
- Return values and multiple returns
- Lambda functions (like JS arrow functions)
- Real-world: API clients, data processors, utilities

### 5. **Error Handling** (`05_error_handling.py`)
- try/except (vs Go's error returns, JS try/catch)
- Exception types and custom exceptions
- finally and resource cleanup
- When to raise vs return error
- Real-world: API calls, file operations, validation

### 6. **Working with Files** (`06_files_and_io.py`)
- Reading/writing files (context managers)
- JSON, CSV, text files
- Path handling (vs Go's filepath, Node's path)
- Real-world: Config files, data processing, logs

### 7. **String Operations** (`07_strings.py`)
- String methods and formatting
- f-strings (better than Go's fmt.Sprintf)
- Template strings (like JS template literals)
- Common string operations
- Real-world: Log parsing, text processing, formatting

### 8. **Working with APIs** (`08_http_and_apis.py`)
- Making HTTP requests (sync version)
- requests library (like axios in JS)
- Handling responses and errors
- JSON parsing
- Real-world: Calling OpenAI, GitHub, AWS APIs

## 🏃 Quick Start

```bash
# Run any example
poetry run python 00-python-essentials/examples/01_variables_and_types.py

# Or activate the environment first
poetry shell
python 00-python-essentials/examples/01_variables_and_types.py
```

## 💡 Teaching Approach

Each example follows this pattern:

1. **Concept** - What it is and why it matters
2. **Go/JS Comparison** - How it relates to what you know
3. **Basic Example** - Simple demonstration
4. **Real-World Example** - Practical use case from DevOps/ML
5. **Common Pitfalls** - Gotchas to avoid
6. **Best Practices** - The Pythonic way

## 🔄 Relationship to Other Languages

```
Go                  →  Python              →  Use Case
-----------------      ------------------     -----------------------
fmt.Println()       →  print()             →  Simpler, dynamic
if err != nil       →  try/except          →  Exception-based
make([]string, 0)   →  []                  →  Dynamic sizing
map[string]int      →  dict[str, int]      →  Same concept
struct              →  class/dataclass     →  More flexible

JavaScript          →  Python              →  Use Case
-----------------      ------------------     -----------------------
console.log()       →  print()             →  Similar
const/let           →  (just use names)    →  No const/let
array.map()         →  list comprehension  →  More powerful
{...obj}            →  {**dict}            →  Similar spreading
async/await         →  async/await         →  Very similar!
```

## ✅ Prerequisites

- Experience with Go, JavaScript, or similar language
- Basic understanding of programming concepts
- DevOps/infrastructure experience helpful

## 🎓 After This Module

After completing this module, you'll be ready for:
- **Module 1**: Python Fundamentals (decorators, context managers)
- **Module 2**: Async Python (for concurrent API calls)
- **Module 3**: Data Structures (NumPy, Pandas)

## 📖 Learning Tips

1. **Run the examples** - Don't just read them
2. **Modify the code** - Change values and see what happens
3. **Compare to Go/JS** - Think about how you'd do this in those languages
4. **Think about your work** - How would you use this in your projects?

## 🔗 Next Steps

Start with `01_variables_and_types.py` and work through sequentially. Each example builds on the previous one.
