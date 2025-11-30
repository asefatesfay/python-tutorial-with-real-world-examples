# Module 1: Python Fundamentals for Senior Engineers

> Skip the basics. Focus on what makes Python different from Go and JavaScript.

## 🎯 Learning Objectives

By the end of this module, you'll understand:
- Python's type system and type hints (vs Go's static typing)
- Decorators and metaprogramming (more powerful than JS decorators)
- Context managers for resource management
- Generators and comprehensions (memory-efficient iterations)
- Python's data model and magic methods
- Module system and imports

## 📋 Table of Contents

1. [Type Hints & Static Analysis](#1-type-hints--static-analysis)
2. [Decorators & Metaprogramming](#2-decorators--metaprogramming)
3. [Context Managers](#3-context-managers)
4. [Comprehensions & Generators](#4-comprehensions--generators)
5. [Data Model & Magic Methods](#5-data-model--magic-methods)
6. [Module System](#6-module-system)

---

## 1. Type Hints & Static Analysis

### Coming from Go
Go has compile-time type checking. Python is dynamic but supports optional type hints.

**Go:**
```go
func processData(data []string, limit int) ([]string, error) {
    // Type safety at compile time
}
```

**Python:**
```python
from typing import List, Optional

def process_data(data: list[str], limit: int) -> list[str]:
    # Type hints for tooling (mypy, IDEs)
    # Runtime doesn't enforce types!
    return data[:limit]
```

### Key Concepts

- **Type hints are optional** - for tooling, not runtime
- **Use mypy** for static type checking in CI/CD
- **Generic types** - `list[T]`, `dict[K, V]`, `Optional[T]`
- **Union types** - `str | int` (Python 3.10+)
- **Type aliases** - for complex types

### Examples

See: [`examples/01_type_hints.py`](./examples/01_type_hints.py)

---

## 2. Decorators & Metaprogramming

### Coming from JavaScript
JS has decorators (proposal), but Python's are more mature and powerful.

**JavaScript (TypeScript):**
```typescript
function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    // Decorator syntax
}

class MyClass {
    @log
    method() {}
}
```

**Python:**
```python
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log
def my_function():
    pass
```

### Key Concepts

- **Function decorators** - modify functions
- **Class decorators** - modify classes
- **Parameterized decorators** - decorators with arguments
- **Built-in decorators** - `@property`, `@staticmethod`, `@classmethod`
- **functools.wraps** - preserve metadata

### Real-World Use Cases for AI/ML

- Timing/profiling ML model inference
- Caching expensive embeddings
- Retry logic for API calls
- Authentication/authorization
- Logging and monitoring

### Examples

See: [`examples/02_decorators.py`](./examples/02_decorators.py)

---

## 3. Context Managers

### Coming from Go
Similar to `defer`, but more structured.

**Go:**
```go
file, err := os.Open("file.txt")
if err != nil {
    return err
}
defer file.Close()
```

**Python:**
```python
with open("file.txt") as file:
    # Automatically closed when exiting block
    data = file.read()
```

### Key Concepts

- **`with` statement** - automatic resource cleanup
- **`__enter__` and `__exit__`** - context manager protocol
- **`contextlib`** - helpers for creating context managers
- **Async context managers** - `async with`

### Real-World Use Cases for AI/ML

- Database connections
- Vector database clients
- File I/O for large datasets
- Temporary directories for model artifacts
- GPU memory management

### Examples

See: [`examples/03_context_managers.py`](./examples/03_context_managers.py)

---

## 4. Comprehensions & Generators

### Coming from JavaScript
More powerful than JS map/filter/reduce, more memory-efficient.

**JavaScript:**
```javascript
const squared = numbers.map(x => x * x).filter(x => x > 10);
// Creates intermediate arrays
```

**Python:**
```python
# List comprehension
squared = [x * x for x in numbers if x * x > 10]

# Generator expression (lazy, memory-efficient)
squared_gen = (x * x for x in numbers if x * x > 10)
```

### Key Concepts

- **List comprehensions** - `[expr for item in iterable if condition]`
- **Dict comprehensions** - `{k: v for item in iterable}`
- **Set comprehensions** - `{expr for item in iterable}`
- **Generator expressions** - `(expr for item in iterable)` (lazy)
- **Generator functions** - `yield` keyword

### Real-World Use Cases for AI/ML

- Processing large datasets without loading all into memory
- Streaming document chunks for embeddings
- Batch processing for model inference
- Data transformation pipelines

### Examples

See: [`examples/04_comprehensions_generators.py`](./examples/04_comprehensions_generators.py)

---

## 5. Data Model & Magic Methods

### Coming from Go
Similar to implementing interfaces, but more flexible.

**Go:**
```go
type Stringer interface {
    String() string
}
```

**Python:**
```python
class MyClass:
    def __str__(self):
        return "String representation"
    
    def __repr__(self):
        return "MyClass()"
```

### Key Concepts

- **Dunder methods** - `__method__` (double underscore)
- **`__init__`** - constructor (like `constructor` in JS)
- **`__str__` vs `__repr__`** - string representations
- **`__len__`, `__getitem__`** - make objects behave like sequences
- **`__call__`** - make objects callable like functions
- **`__enter__`, `__exit__`** - context manager protocol

### Real-World Use Cases for AI/ML

- Custom dataset classes for ML training
- Embeddings wrapper classes
- Custom vector store implementations
- Model wrapper classes

### Examples

See: [`examples/05_data_model.py`](./examples/05_data_model.py)

---

## 6. Module System

### Coming from Go & JavaScript

**Go:**
```go
import "github.com/user/package"
```

**JavaScript:**
```javascript
import { something } from './module.js';
const module = require('./module');
```

**Python:**
```python
# Absolute import
from package.module import something

# Relative import (within package)
from .sibling import helper
from ..parent import config
```

### Key Concepts

- **Modules** - any `.py` file
- **Packages** - directories with `__init__.py`
- **Absolute vs relative imports**
- **`__name__ == "__main__"`** - script entry point
- **`sys.path`** - import search paths
- **Namespace packages** - PEP 420

### Best Practices

- Use absolute imports for clarity
- Avoid circular imports
- Organize code into packages
- Use `if __name__ == "__main__":` for scripts

### Examples

See: [`examples/06_modules/`](./examples/06_modules/)

---

## 🏗️ Mini Project: LLM Response Cache

Build a decorator-based caching system for LLM API calls (useful for development).

**Features:**
- Decorator to cache function results
- Context manager for cache lifecycle
- Type hints for all functions
- Generator for streaming responses

**Location:** [`project/`](./project/)

---

## 🎯 Exercises

Test your understanding with practical exercises:

1. **Type Hints** - Add comprehensive type hints to untyped code
2. **Decorator** - Build a retry decorator for API calls
3. **Context Manager** - Create a timer context manager
4. **Generator** - Build a batch generator for processing large datasets
5. **Magic Methods** - Create a custom embedding vector class

**Location:** [`exercises/`](./exercises/)

---

## 🔑 Key Takeaways

1. **Type hints are optional but valuable** - use mypy in CI/CD
2. **Decorators are powerful** - use for cross-cutting concerns
3. **Context managers ensure cleanup** - use for resources
4. **Generators are memory-efficient** - use for large data
5. **Magic methods enable Pythonic APIs** - make objects intuitive
6. **Imports are flexible** - organize code well

---

## ⏭️ Next Module

[Module 2: Async Python](../02-async-python/) - Learn asyncio patterns for parallel API calls and data processing.

---

## 📚 Additional Resources

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 343 - The "with" Statement](https://peps.python.org/pep-0343/)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Real Python - Decorators](https://realpython.com/primer-on-python-decorators/)
