# Contributing to Python Mastery Tutorial

## 🤝 Welcome!

This is your personal learning journey, but you're welcome to:
- Report issues you find
- Suggest improvements
- Share your solutions to exercises
- Add new examples or modules

## 📝 Reporting Issues

Found a bug or unclear explanation?

1. Check if issue already exists
2. Create new issue with:
   - Module and file name
   - What you expected vs what happened
   - Your Python version and OS
   - Code snippet if applicable

## 💡 Suggesting Enhancements

Have ideas for improvements?

- New examples or use cases
- Better explanations
- Additional comparisons to Go/JS
- More AI/ML relevant examples

## 🔧 Contributing Code

### Setup Development Environment

```bash
# Clone the repo
git clone <your-fork>
cd python-tutorial-with-real-world-examples

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies including dev dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Activate Poetry shell for development
poetry shell
```

### Code Style

We follow:
- **PEP 8** - Python style guide
- **Type hints** - All functions should have type hints
- **Docstrings** - Google style docstrings
- **Black** - Code formatting
- **mypy** - Static type checking

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Check types
poetry run mypy .

# Lint
poetry run pylint src/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

### Adding New Examples

1. Create example file in appropriate module
2. Follow this structure:

```python
"""
Example Name - Brief Description

Detailed description of what this example demonstrates.

Run: python example_name.py
"""

from typing import ...

# ============================================================================
# Section 1: Concept Name
# ============================================================================

def demo_concept():
    """Demonstrate concept with clear output."""
    # Implementation
    pass


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Example Name ===\n")
    demo_concept()
    print("\n✅ Key takeaway message")


if __name__ == "__main__":
    main()
```

3. Add to module README
4. Test thoroughly

### Adding New Modules

1. Create module directory: `XX-module-name/`
2. Create `README.md` with:
   - Learning objectives
   - Comparisons to Go/JS
   - Concepts covered
   - Examples list
3. Create `examples/` directory
4. Create `exercises/` directory
5. Create `project/` directory
6. Update main README

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_module.py

# With coverage (configured in pyproject.toml)
poetry run pytest --cov

# Run tests in watch mode
poetry run pytest-watch
```

## 📚 Documentation

- Keep explanations concise
- Include comparisons to Go/JS where relevant
- Focus on AI/ML use cases
- Add code comments for complex logic
- Include expected output in examples

## 🎨 Example Quality Checklist

- [ ] Type hints on all functions
- [ ] Clear docstrings
- [ ] Runnable standalone
- [ ] Prints clear output
- [ ] Demonstrates one concept well
- [ ] Includes AI/ML context
- [ ] Compares to Go/JS if applicable
- [ ] Has comments for complex parts
- [ ] Follows naming conventions

## 🐛 Bug Fix Process

1. Write test that reproduces bug
2. Fix the bug
3. Ensure test passes
4. Update documentation if needed

## 📖 Documentation Updates

- Fix typos and grammatical errors
- Improve unclear explanations
- Add missing information
- Update outdated references

## ⚡ Quick Contribution Guide

### Small Changes (Typos, Fixes)
1. Fork repo
2. Make changes
3. Submit PR

### Large Changes (New Modules)
1. Open issue first to discuss
2. Get feedback on approach
3. Implement
4. Submit PR

## 🎓 Learning Path Principles

When contributing, keep these principles:

1. **No Beginner Fluff** - Assume programming knowledge
2. **Leverage Experience** - Compare to Go/JS patterns
3. **AI/ML Focus** - Examples must be relevant
4. **Production Ready** - Show best practices
5. **Hands-On** - Build real applications

## 📬 Contact

Questions or need help?
- Open an issue
- Tag me in discussions
- Share in the community

## 🙏 Thank You!

Every contribution makes this tutorial better for everyone learning Python for AI/ML engineering.

Happy coding! 🐍🚀
