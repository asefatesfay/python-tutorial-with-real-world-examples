# Poetry Configuration and Usage Guide

## 🎯 Why Poetry?

Poetry is the modern standard for Python package management. Benefits:

- ✅ **Dependency resolution** - Handles conflicts automatically
- ✅ **Lock file** - Reproducible builds across environments
- ✅ **Virtual environment** - Automatic creation and management
- ✅ **Build & publish** - Package distribution made easy
- ✅ **Dev dependencies** - Separate dev/prod dependencies
- ✅ **Scripts** - Define custom commands
- ✅ **pyproject.toml** - Single source of truth

## 📦 Installation

### macOS/Linux
```bash
# Recommended: Official installer
curl -sSL https://install.python-poetry.org | python3 -

# Or using Homebrew
brew install poetry

# Or using pipx
pipx install poetry
```

### Windows
```powershell
# PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Verify Installation
```bash
poetry --version
# Poetry (version 1.7.0)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install all dependencies (creates virtual environment)
poetry install

# Install only production dependencies
poetry install --only main

# Install with dev dependencies
poetry install --with dev

# Install with optional docs dependencies
poetry install --with docs
```

### 2. Activate Virtual Environment
```bash
# Option 1: Activate shell
poetry shell

# Now you can run commands directly
python --version
pytest
mypy .

# Exit shell
exit

# Option 2: Run commands without activating
poetry run python script.py
poetry run pytest
poetry run mypy .
```

### 3. Add New Dependencies
```bash
# Add production dependency
poetry add requests

# Add dev dependency
poetry add --group dev pytest-mock

# Add with version constraint
poetry add "numpy>=1.24.0,<2.0.0"

# Add from git
poetry add git+https://github.com/user/repo.git
```

### 4. Remove Dependencies
```bash
# Remove package
poetry remove requests

# Remove dev dependency
poetry remove --group dev pytest-mock
```

### 5. Update Dependencies
```bash
# Update all dependencies
poetry update

# Update specific package
poetry update numpy

# Show outdated packages
poetry show --outdated
```

## 📋 Common Commands

### Environment Management
```bash
# Show virtual environment info
poetry env info

# List all virtual environments
poetry env list

# Remove virtual environment
poetry env remove python

# Use specific Python version
poetry env use python3.11
poetry env use /usr/local/bin/python3.11
```

### Dependency Management
```bash
# List installed packages
poetry show

# List only production packages
poetry show --only main

# Show package details
poetry show numpy

# Show dependency tree
poetry show --tree

# Export requirements.txt (for compatibility)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Development Workflow
```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov

# Type checking
poetry run mypy .

# Code formatting
poetry run black .

# Import sorting
poetry run isort .

# Linting
poetry run pylint src/

# All quality checks
poetry run pre-commit run --all-files
```

### Build & Package
```bash
# Build package (wheel and sdist)
poetry build

# Publish to PyPI
poetry publish

# Publish to test PyPI
poetry publish -r testpypi
```

## 🎓 Project Structure with Poetry

```
python-tutorial-with-real-world-examples/
├── pyproject.toml           # Poetry configuration & dependencies
├── poetry.lock              # Lock file (commit this!)
├── .venv/                   # Virtual environment (auto-created)
├── README.md
├── src/                     # Source code
│   └── package/
│       ├── __init__.py
│       └── module.py
├── tests/                   # Tests
│   └── test_module.py
└── docs/                    # Documentation
```

## ⚙️ Configuration

### Project Settings (pyproject.toml)

```toml
[tool.poetry]
name = "python-mastery-tutorial"
version = "0.1.0"
description = "Python tutorial"
authors = ["Your Name <email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.24.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
mypy = "^1.7.0"
```

### Global Poetry Settings

```bash
# Configure virtual environment location
poetry config virtualenvs.in-project true  # Create .venv in project
poetry config virtualenvs.in-project false # Create in cache dir

# List all settings
poetry config --list

# Show cache directory
poetry config cache-dir
```

## 🔧 Working with This Tutorial

### Daily Workflow
```bash
# 1. Start working
cd python-tutorial-with-real-world-examples
poetry shell

# 2. Run examples
cd 01-python-fundamentals/examples
python 01_type_hints.py

# 3. Make changes, format code
black my_script.py

# 4. Run tests
pytest

# 5. Exit
exit
```

### Adding New Features
```bash
# Need a new library?
poetry add chromadb

# Need it only for development?
poetry add --group dev ipython

# Update lock file
git add poetry.lock pyproject.toml
git commit -m "Add chromadb dependency"
```

### Updating Dependencies
```bash
# Check for updates
poetry show --outdated

# Update specific package
poetry update langchain

# Update all
poetry update

# Commit the updated lock file
git add poetry.lock
git commit -m "Update dependencies"
```

## 🎯 Poetry vs pip/venv

| Feature | Poetry | pip + venv |
|---------|--------|------------|
| Dependency resolution | ✅ Automatic | ❌ Manual |
| Lock file | ✅ poetry.lock | ❌ No |
| Virtual env | ✅ Automatic | ❌ Manual |
| Dev dependencies | ✅ Separate | ⚠️ requirements-dev.txt |
| Build packages | ✅ Built-in | ❌ Need setuptools |
| Publish packages | ✅ Built-in | ❌ Need twine |
| Configuration | ✅ pyproject.toml | ⚠️ Multiple files |

## 💡 Pro Tips

### 1. Always Commit poetry.lock
```bash
git add poetry.lock pyproject.toml
git commit -m "Update dependencies"
```

### 2. Use Poetry Scripts
```toml
[tool.poetry.scripts]
train = "src.training:main"
serve = "src.api:run_server"
```

```bash
poetry run train
poetry run serve
```

### 3. Export requirements.txt for CI/CD
```bash
# For Docker or systems that use pip
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### 4. Speed Up Installation
```bash
# Skip dev dependencies in production
poetry install --only main --no-root

# Use parallel installation
poetry install --no-interaction --parallel 4
```

### 5. Work with Jupyter
```bash
# Install Jupyter in Poetry environment
poetry add --group dev jupyter ipykernel

# Create kernel
poetry run python -m ipykernel install --user --name=tutorial-env

# Start Jupyter
poetry run jupyter notebook
```

## 🆘 Troubleshooting

### Issue: Poetry command not found
```bash
# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: Lock file is out of date
```bash
# Update lock file
poetry lock --no-update

# Or force recreate
rm poetry.lock
poetry install
```

### Issue: Dependency conflicts
```bash
# Show conflict details
poetry add package-name --dry-run

# Update conflicting package
poetry update conflicting-package
```

### Issue: Virtual environment issues
```bash
# Remove and recreate
poetry env remove python
poetry install
```

## 📚 Learn More

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry Basic Usage](https://python-poetry.org/docs/basic-usage/)
- [Dependency Specification](https://python-poetry.org/docs/dependency-specification/)
- [Managing Environments](https://python-poetry.org/docs/managing-environments/)

---

**Ready to use Poetry?** Run `poetry install` to get started! 🚀
