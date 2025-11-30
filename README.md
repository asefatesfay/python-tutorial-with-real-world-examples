# Python Mastery for ML/AI Engineering

**A comprehensive, hands-on tutorial series designed for senior engineers (Go/JS/DevOps background) transitioning to Python for Machine Learning and AI.**

**Philosophy**: Learn by building. Understand concepts through real-world examples. Ship production-ready code.

---

## 🎯 Who Is This For?

- **Senior engineers** from Go, JavaScript, or similar backgrounds
- Want to master Python for **ML/AI engineering** (not research)
- Prefer **practical examples** over theoretical lectures
- Need to ship **production ML/AI systems**
- Want to work with **LLMs, RAG, embeddings, and modern AI**

**Not for**: Complete programming beginners, AI researchers, pure theoreticians

---

## 📚 Complete Curriculum

### **Phase 1: Python Foundations** (Modules 0-2)
Master Python essentials before diving into ML/AI.

#### [Module 0: Python Essentials](00-python-essentials/) 
*Fundamentals for engineers coming from other languages*
- Variables, types, collections, control flow
- Functions, error handling, files, strings
- HTTP and APIs
- **8 examples** with Go/JS comparisons

#### [Module 1: Python Fundamentals](01-python-fundamentals/)
*Advanced Python patterns for ML engineering*
- Decorators, context managers, generators
- Itertools, functools, dataclasses
- Comprehensions, lambdas, type hints
- **5 examples + mini-project**

#### [Module 2: Async Python](02-async-python/)
*Concurrent programming for APIs and data processing*
- async/await, asyncio
- Parallel API calls, rate limiting
- Streaming LLM responses
- **4 examples**

---

### **Phase 2: Mathematical Foundations** (Module 3)
Build intuition for the math behind ML/AI.

#### [Module 3: Math for ML/AI](03-math-for-ml/)
*Essential math with intuitive explanations and Python code*

**Statistics**: Mean, variance, distributions, Bayes' theorem, correlation
**Linear Algebra**: Vectors, matrices, dot product, eigenvalues
**Calculus**: Derivatives, gradient descent, backpropagation
**Information Theory**: Entropy, cross-entropy, loss functions

**4 comprehensive examples** - no proofs, all intuition!

---

### **Phase 3: Data Engineering** (Modules 4-5)
Master efficient data manipulation and feature engineering.

#### [Module 4: NumPy & Pandas](04-numpy-pandas/)
*Fast numerical computing and data manipulation*
- NumPy arrays and vectorization (10-100x faster)
- Pandas DataFrames and operations
- Data loading, cleaning, transformation
- **6 examples**

#### [Module 5: Feature Engineering](05-feature-engineering/)
*Transform raw data into ML-ready features*
- Numerical transformations and encoding
- Categorical encoding techniques
- Text and datetime features
- Feature selection and dimensionality reduction
- **7 examples + project**

---

### **Phase 4: Classical Machine Learning** (Module 6)
Build ML fundamentals from scratch and with scikit-learn.

#### [Module 6: ML Fundamentals](06-machine-learning-fundamentals/)
*Supervised and unsupervised learning*
- Linear/Logistic Regression, Decision Trees
- Random Forests, Gradient Boosting
- K-Means, PCA, Anomaly Detection
- Model evaluation and hyperparameter tuning
- **12 examples + 3 projects**

---

### **Phase 5: Deep Learning** (Module 7)
Neural networks with PyTorch for modern AI.

#### [Module 7: Deep Learning with PyTorch](07-deep-learning/)
*Build and train neural networks*
- PyTorch fundamentals (tensors, autograd)
- CNNs for computer vision
- RNNs/Transformers for NLP
- Transfer learning, data augmentation
- **15+ examples + 3 projects**

---

### **Phase 6: LLMs and Modern AI** (Modules 8-9)
Work with Large Language Models and build RAG systems.

#### [Module 8: LLMs and Embeddings](08-llms-and-embeddings/)
*OpenAI, Anthropic, and vector databases*
- LLM APIs (GPT-4, Claude)
- Prompt engineering
- Text embeddings and similarity search
- Vector databases (ChromaDB, Pinecone)
- **15 examples + 3 projects**

#### [Module 9: RAG and LangChain](09-rag-langchain/)
*Retrieval Augmented Generation for Q&A over your data*
- LangChain framework
- Document loading and chunking
- RAG pipelines with memory
- Advanced retrieval techniques
- RAG evaluation
- **15 examples + 3 projects**

---

### **Phase 7: Production Deployment** (Module 10)
Ship ML/AI to production with monitoring and scaling.

#### [Module 10: Deployment & Production](10-deployment-production/)
*MLOps and production ML systems*
- FastAPI for ML serving
- Docker and containerization
- Cloud deployment (AWS, GCP, Azure)
- Monitoring, logging, alerting
- CI/CD for ML
- Cost optimization
- **20+ examples + 3 production projects**

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/python-tutorial-with-real-world-examples.git
cd python-tutorial-with-real-world-examples

# Install Poetry (package manager)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Start with Module 0
cd 00-python-essentials/examples
poetry run python 01_variables_and_types.py
```

---

## 📖 How to Use This Tutorial

### **Option 1: Sequential Learning** (Recommended)
Work through modules 0 → 10 in order. Each builds on previous concepts.

**Timeline**: 3-6 months (2-3 hours/day)

### **Option 2: Need-Based Learning**
Jump to modules relevant to your current work:
- Building APIs? → Modules 0, 2, 10
- Feature engineering? → Modules 3, 4, 5
- Deep learning? → Modules 3, 6, 7
- LLM applications? → Modules 0, 8, 9, 10

### **Option 3: Quick Reference**
Use as documentation when you encounter specific concepts in your work.

---

## 💡 What Makes This Different?

| Traditional Courses | This Tutorial |
|---------------------|---------------|
| Theory-heavy lectures | Hands-on code you can run |
| Toy datasets | Real-world scenarios |
| Academic focus | Production-ready patterns |
| Generic examples | DevOps/ML-specific |
| One language | Compares Python to Go/JS |
| Abstract concepts | Concrete implementations |

---

## 🎯 Learning Outcomes

After completing this tutorial, you'll be able to:

✅ **Write Pythonic code** that senior Python devs respect  
✅ **Understand ML math** (linear algebra, calculus, statistics)  
✅ **Build ML models** from scratch and with scikit-learn  
✅ **Train neural networks** with PyTorch  
✅ **Work with LLMs** (GPT-4, Claude) via APIs  
✅ **Build RAG systems** for Q&A over your data  
✅ **Deploy to production** with FastAPI, Docker, cloud  
✅ **Monitor and optimize** ML systems at scale  

---

## 🔧 Prerequisites

**Required**:
- Programming experience (Go, JavaScript, or similar)
- Command line comfort
- Basic understanding of APIs

**Helpful but not required**:
- High school math (we'll teach the rest!)
- AWS/GCP experience
- Docker knowledge

---

## 📦 Dependencies

**Core**:
```bash
poetry add numpy pandas matplotlib seaborn
poetry add scikit-learn scipy
poetry add torch torchvision torchaudio
poetry add openai anthropic langchain
poetry add fastapi uvicorn
```

**Full list**: See `pyproject.toml`

---

## 🗺️ Module Dependencies

```
Module 0 (Python Essentials)
   ↓
Module 1 (Python Fundamentals)
   ↓
Module 2 (Async Python)
   ↓
Module 3 (Math for ML) ←────────┐
   ↓                             │
Module 4 (NumPy/Pandas)         │
   ↓                             │
Module 5 (Feature Engineering)  │
   ↓                             │
Module 6 (ML Fundamentals) ─────┤
   ↓                             │
Module 7 (Deep Learning) ───────┤
   ↓                             │
Module 8 (LLMs/Embeddings) ─────┘
   ↓
Module 9 (RAG/LangChain)
   ↓
Module 10 (Production Deployment)
```

---

## 🏆 Projects Included

Each phase includes real-world projects:

**Phase 2**: Simple web scraper with async
**Phase 5**: Customer churn prediction
**Phase 6**: House price prediction, spam classifier, customer segmentation
**Phase 7**: CIFAR-10 classifier, sentiment analyzer, recommendation system
**Phase 8**: Semantic search engine, chatbot
**Phase 9**: Documentation Q&A, research assistant
**Phase 10**: Production ML API, RAG API with streaming

---

## 📈 Difficulty Progression

```
Module 0-2:  ⭐⭐☆☆☆  Python fundamentals
Module 3:    ⭐⭐⭐☆☆  Math intuition
Module 4-5:  ⭐⭐⭐☆☆  Data engineering
Module 6:    ⭐⭐⭐⭐☆  Classical ML
Module 7:    ⭐⭐⭐⭐☆  Deep learning
Module 8-9:  ⭐⭐⭐⭐☆  Modern AI (LLMs, RAG)
Module 10:   ⭐⭐⭐⭐⭐  Production systems
```

---

## 🤝 Contributing

Found a bug? Have a suggestion? 

1. Open an issue
2. Submit a pull request
3. Share your projects built with this tutorial!

---

## 📝 License

MIT License - Use freely for learning and commercial projects.

---

## 🙏 Acknowledgments

Inspired by real-world ML/AI engineering work and the need for practical, production-focused education.

Built with insights from:
- Production ML systems at scale
- Common pain points of engineers learning ML
- Best practices from senior ML engineers
- Modern AI tooling (OpenAI, LangChain, PyTorch)

---

## 📬 Stay Updated

- **GitHub**: Star this repo for updates
- **Issues**: Ask questions, request topics
- **Discussions**: Share your projects and learnings

---

## 🚀 Ready to Begin?

```bash
# Start your journey
cd 00-python-essentials
poetry run python examples/01_variables_and_types.py

# Or jump to what you need
cd 08-llms-and-embeddings  # For LLM work
cd 10-deployment-production  # For MLOps
```

---

## 📚 Additional Resources

**Books**:
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with PyTorch" by Stevens et al.
- "Designing Data-Intensive Applications" by Kleppmann

**Online**:
- Fast.ai courses (practical deep learning)
- PyTorch tutorials (official docs)
- LangChain documentation
- Papers With Code (latest research)

**Practice**:
- Kaggle competitions
- LeetCode (Python interview prep)
- GitHub open source ML projects

---

**Happy Learning! Build something amazing.** 🚀🐍🤖