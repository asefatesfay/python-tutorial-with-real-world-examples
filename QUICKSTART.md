# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Python Environment with Poetry

```bash
# Navigate to the tutorial directory
cd python-tutorial-with-real-world-examples

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Or using homebrew on macOS
brew install poetry

# Configure Poetry to create virtual environments in the project
poetry config virtualenvs.in-project true

# Install all dependencies (creates virtual environment automatically)
poetry install

# Activate the virtual environment
poetry shell

# Or run commands without activating shell
poetry run python --version
```

**Alternative: Using pip (if you prefer)**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies from requirements.txt (generated from Poetry)
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check Python version (should be 3.10+)
poetry run python --version

# Check Poetry installation
poetry --version

# Run type checker
poetry run mypy --version

# Run a sample program
poetry run python 01-python-fundamentals/examples/01_type_hints.py
```

### 3. Set Up API Keys (for AI/ML modules)

Create a `.env` file in the root directory:

```bash
# OpenAI API Key (for embeddings and LLM calls)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
```

Get your API keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- Pinecone: https://app.pinecone.io/

### 4. Run Your First Example

```bash
# Navigate to Module 1
cd 01-python-fundamentals/examples

# Run type hints example
poetry run python 01_type_hints.py

# Run decorators example
poetry run python 02_decorators.py

# Run all examples (in Poetry shell)
poetry shell
python 01_type_hints.py
python 02_decorators.py
python 03_context_managers.py
python 04_comprehensions_generators.py
python 05_data_model.py
exit  # Exit poetry shell
```

### 5. Follow the Learning Path

Start with **Module 1** and work your way through:

```bash
# Week 1: Python Fundamentals
01-python-fundamentals/

# Week 2: Async & Data
02-async-python/
03-data-structures/

# Week 3: APIs & ML Serving
04-fastapi-ml-serving/

# Week 4-5: AI/ML Core
05-embeddings-vectors/
06-langchain-basics/
07-rag-applications/

# Week 6: Production
08-testing-best-practices/
09-deployment-mlops/
10-capstone-project/
```

## 🎓 Learning Tips

### For Senior Engineers

1. **Skim the basics** - Focus on Python-specific patterns
2. **Compare to Go/JS** - Each module has comparisons
3. **Run the code** - Don't just read, execute and modify
4. **Build projects** - Each module has a mini-project
5. **Focus on AI/ML** - All examples are relevant to your goal

### Suggested Pace

- **Fast track (1-2 weeks)**: Focus on examples, skip exercises
- **Normal pace (3-4 weeks)**: Do all examples and exercises
- **Thorough (5-6 weeks)**: Complete all projects and capstone

### Study Routine

```
Each module (1-2 days):
├── 30 min: Read README
├── 60 min: Run and study examples
├── 30 min: Try exercises
└── 60 min: Build mini-project
```

## 🛠️ IDE Setup (VS Code Recommended)

### Install Extensions

1. **Python** (Microsoft) - Python language support
2. **Pylance** - Fast Python language server
3. **Python Type Hint** - Type hints support
4. **Python Docstring Generator** - Auto-generate docstrings
5. **Better Comments** - Highlight TODOs and notes

### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

## 🔍 Debugging Tips

### Run with debugging

```bash
# Print debugging
python -i script.py  # Interactive mode after execution

# Use IPython for better REPL
ipython

# Use pdb debugger
python -m pdb script.py
```

### Check Types

```bash
# Check types with mypy
mypy 01-python-fundamentals/examples/01_type_hints.py

# Check all Python files
mypy .
```

## 📚 Next Steps

1. **Complete Module 1** - Python fundamentals
2. **Set up your environment** - Install all dependencies
3. **Join the community** - Share your progress
4. **Build something** - Apply what you learn immediately

## 🆘 Common Issues

### Issue: `ModuleNotFoundError`

```bash
# Make sure you're using Poetry
poetry install

# Or activate the Poetry shell
poetry shell

# Check which Python is being used
poetry run which python
```

### Issue: Type checking errors

```bash
# Type stubs are already included in Poetry dev dependencies
poetry install --with dev

# Run type checker
poetry run mypy .

# Ignore specific errors in code
# type: ignore
```

### Issue: API rate limits

- Use caching decorators (covered in Module 1)
- Implement retry logic
- Consider using local models (Ollama) for development

## 💡 Pro Tips

1. **Use type hints from day one** - Better IDE support
2. **Profile your code** - Use the timer decorator
3. **Test incrementally** - Don't wait until the end
4. **Read the module READMEs** - They have context and comparisons
5. **Modify the examples** - Best way to learn

---

**Ready to start?** Head to [Module 1: Python Fundamentals](./01-python-fundamentals/)

Happy coding! 🐍🚀
