"""
Docker for ML Model Deployment

Learn how to containerize ML models for consistent deployment.
Focus: Docker fundamentals, best practices, and ML-specific patterns.

Note: This file explains Docker concepts with examples.
      Actual Dockerfiles are in the same directory.
"""

import os
from typing import Dict, List


# ============================================================================
# 1. Why Docker for ML?
# ============================================================================

def demo_why_docker():
    """
    Why Docker is essential for ML deployment.
    
    INTUITION - The Shipping Container Analogy:
    
    Before Containers (1950s):
    - Loading ship = nightmare
    - Different cargo → Different handling
    - Delays, damage, expensive
    
    After Containers:
    - Standard box (20ft or 40ft)
    - Same handling everywhere
    - Fast, reliable, cheap
    
    Same for Software:
    
    Before Docker:
    - "Works on my machine" ← Famous last words
    - Different Python versions
    - Missing dependencies
    - Environment chaos
    
    After Docker:
    - Package everything (code + dependencies)
    - Same environment everywhere
    - Dev = Prod (no surprises!)
    - Fast, reliable deployment
    
    WHY DOCKER FOR ML:
    
    1. Reproducibility:
       Your laptop: Python 3.11, CUDA 12.0, PyTorch 2.1
       Colleague: Python 3.9, CUDA 11.8, PyTorch 2.0
       Production: Python 3.10, CUDA 12.1, PyTorch 2.2
       
       Result: Model breaks! ❌
       
       With Docker: Same environment everywhere ✅
    
    2. Dependency Isolation:
       Without Docker:
       - Model A needs TensorFlow 2.14
       - Model B needs TensorFlow 2.10
       - Conflict! Can't install both
       
       With Docker:
       - Model A container: TensorFlow 2.14
       - Model B container: TensorFlow 2.10
       - Both run happily! ✅
    
    3. Easy Deployment:
       Without Docker:
       1. Install Python
       2. Install CUDA
       3. Install dependencies
       4. Configure environment
       5. Debug issues (hours!)
       
       With Docker:
       1. docker run my-model
       
       Done! ⚡
    
    4. Scalability:
       Run 1 container or 1000 containers
       Kubernetes orchestrates them
       Auto-scaling based on load
    
    5. Cloud-Agnostic:
       Same Docker image runs on:
       - AWS (ECS, EKS, Fargate)
       - GCP (Cloud Run, GKE)
       - Azure (ACI, AKS)
       - Your laptop
       
       No vendor lock-in!
    
    TYPICAL ML DOCKER WORKFLOW:
    
    Step 1: Develop Locally
    - Write code
    - Train model
    - Test predictions
    
    Step 2: Dockerize
    - Create Dockerfile
    - docker build -t my-model:v1
    - docker run -p 8000:8000 my-model:v1
    
    Step 3: Test Locally
    - curl http://localhost:8000/predict
    - Verify it works
    
    Step 4: Push to Registry
    - docker tag my-model:v1 myregistry/my-model:v1
    - docker push myregistry/my-model:v1
    
    Step 5: Deploy to Cloud
    - Pull image on cloud
    - Run container
    - Done!
    
    DOCKER vs ALTERNATIVES:
    
    Virtual Machines (VMs):
    - Full OS (GBs)
    - Slow to start (minutes)
    - Resource heavy
    
    Docker Containers:
    - Share host OS (MBs)
    - Fast to start (seconds)
    - Lightweight ✅
    
    Conda Environments:
    - Python-only
    - No system dependencies
    - Hard to reproduce
    
    Docker:
    - Everything (Python, libraries, system tools)
    - Perfectly reproducible ✅
    """
    print("=" * 70)
    print("1. Why Docker for ML Deployment?")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Shipping Container Revolution")
    print()
    print("   Before Containers (1950s):")
    print("   🚢 Loading ship:")
    print("      • Bananas → Special handling")
    print("      • Cars → Different handling")
    print("      • Furniture → Another method")
    print("      • Result: Slow, expensive, error-prone")
    print()
    print("   After Standard Containers:")
    print("   📦 Everything in standard boxes (20ft/40ft)")
    print("      • Load with crane (fast)")
    print("      • Same handling everywhere")
    print("      • Result: Shipping revolution! ✅")
    print()
    
    print("🖥️  SAME FOR ML SOFTWARE:")
    print()
    print("   ❌ Without Docker:")
    print("      Developer laptop: Python 3.11, PyTorch 2.1, CUDA 12.0")
    print("      QA server: Python 3.9, PyTorch 2.0, CUDA 11.8")
    print("      Production: Python 3.10, PyTorch 2.2, CUDA 12.1")
    print()
    print('      Result: "Works on my machine!" 🔥')
    print("              (Doesn't work anywhere else)")
    print()
    print("   ✅ With Docker:")
    print("      Everywhere: Same Docker image")
    print("                 (Python 3.11 + PyTorch 2.1 + CUDA 12.0)")
    print()
    print("      Result: Works everywhere! 🎉")
    print()
    
    print("🎯 KEY BENEFITS FOR ML:")
    print()
    print("   1️⃣  Reproducibility:")
    print("      Problem: Model trained on your laptop won't run in prod")
    print("      Solution: Docker freezes exact environment")
    print()
    print("      Before: 'Install Python 3.11, then pip install...'")
    print("              (Different versions → Different results)")
    print()
    print("      After: docker run ml-model:v1")
    print("             (Exact same environment every time)")
    print()
    print("   2️⃣  Dependency Isolation:")
    print("      Problem: Two models need conflicting libraries")
    print()
    print("      Model A: TensorFlow 2.14")
    print("      Model B: TensorFlow 2.10")
    print()
    print("      Without Docker: Can't install both! ❌")
    print("      With Docker: Each in own container ✅")
    print()
    print("   3️⃣  Simplified Deployment:")
    print()
    print("      Traditional deployment:")
    print("      1. SSH into server")
    print("      2. Install Python 3.11")
    print("      3. Install CUDA drivers")
    print("      4. pip install 50 dependencies")
    print("      5. Fix conflicts (hours)")
    print("      6. Configure environment vars")
    print("      7. Test (pray it works)")
    print()
    print("      Docker deployment:")
    print("      1. docker run ml-model:v1")
    print()
    print("      Done! ⚡")
    print()
    print("   4️⃣  Cloud-Agnostic:")
    print("      Same Docker image runs on:")
    print("      ☁️  AWS (ECS, EKS, Fargate)")
    print("      ☁️  GCP (Cloud Run, GKE)")
    print("      ☁️  Azure (ACI, AKS)")
    print("      💻 Your laptop")
    print()
    print("      No vendor lock-in! Switch clouds easily.")
    print()
    print("   5️⃣  Scalability:")
    print("      Run 1 container: docker run ml-model")
    print("      Run 1000 containers: kubectl scale --replicas=1000")
    print()
    print("      Same image, different scale!")
    print()
    
    print("📊 DOCKER vs ALTERNATIVES:")
    print()
    print("   Virtual Machines (VMs):")
    print("   • Size: 5-20 GB (full OS)")
    print("   • Start time: 1-5 minutes")
    print("   • Memory: 512MB-2GB overhead")
    print("   • Use case: Full isolation needed")
    print()
    print("   Docker Containers:")
    print("   • Size: 100-500 MB (shared OS)")
    print("   • Start time: 1-5 seconds ⚡")
    print("   • Memory: <100MB overhead")
    print("   • Use case: ML models (perfect!) ✅")
    print()
    print("   Conda Environments:")
    print("   • Size: 1-5 GB")
    print("   • Python libraries only")
    print("   • No system dependencies")
    print("   • Hard to reproduce")
    print("   • Use case: Local development")
    print()
    print("   Winner for ML Deployment: Docker! 🏆")
    print()
    
    print("🔄 TYPICAL ML DOCKER WORKFLOW:")
    print()
    print("   1. Develop Locally:")
    print("      • Write FastAPI code")
    print("      • Train model")
    print("      • Test: python main.py")
    print()
    print("   2. Create Dockerfile:")
    print("      • Define base image (Python)")
    print("      • Install dependencies")
    print("      • Copy code + model")
    print()
    print("   3. Build Image:")
    print("      $ docker build -t ml-model:v1 .")
    print("      → Creates portable image")
    print()
    print("   4. Test Locally:")
    print("      $ docker run -p 8000:8000 ml-model:v1")
    print("      $ curl http://localhost:8000/predict")
    print("      → Verify it works")
    print()
    print("   5. Push to Registry:")
    print("      $ docker push myregistry/ml-model:v1")
    print("      → Available to production")
    print()
    print("   6. Deploy to Cloud:")
    print("      $ kubectl apply -f deployment.yaml")
    print("      → Running in production! 🚀")
    print()
    
    print("💡 REAL-WORLD IMPACT:")
    print()
    print("   Without Docker:")
    print("   • Deployment: 2-4 hours (manual setup)")
    print("   • Bugs: 'Works on my machine' issues")
    print("   • Rollback: 30+ minutes (reinstall old version)")
    print("   • Scaling: Manual (provision servers)")
    print()
    print("   With Docker:")
    print("   • Deployment: 5 minutes (automated)")
    print("   • Bugs: Rare (same environment)")
    print("   • Rollback: 30 seconds (switch image version)")
    print("   • Scaling: Automatic (Kubernetes HPA)")
    print()
    print("   Time saved: ~80% 🎉")


# ============================================================================
# 2. Docker Fundamentals
# ============================================================================

def demo_docker_fundamentals():
    """
    Docker core concepts explained.
    """
    print("\n" + "=" * 70)
    print("2. Docker Fundamentals")
    print("=" * 70)
    print()
    
    print("📦 CORE CONCEPTS:")
    print()
    print("   1. Image (Blueprint)")
    print("   ─────────────────────")
    print("   Think: Cookie cutter 🍪")
    print()
    print("   • Read-only template")
    print("   • Contains:")
    print("     - Base OS (Ubuntu, Alpine, etc.)")
    print("     - Python runtime")
    print("     - Dependencies (pip packages)")
    print("     - Your code")
    print("     - Trained model files")
    print()
    print("   • Created from: Dockerfile")
    print("   • Versioned: ml-model:v1, ml-model:v2")
    print("   • Shareable: Push to Docker Hub")
    print()
    print("   Example:")
    print("   $ docker build -t ml-model:v1 .")
    print("   → Creates image from Dockerfile")
    print()
    
    print("   2. Container (Running Instance)")
    print("   ────────────────────────────────")
    print("   Think: Cookie from cutter 🍪")
    print()
    print("   • Running instance of image")
    print("   • Isolated environment")
    print("   • Has own:")
    print("     - Filesystem")
    print("     - Network")
    print("     - Process space")
    print()
    print("   • Ephemeral: Stop container → Data lost")
    print("   • Multiple containers from same image")
    print()
    print("   Example:")
    print("   $ docker run -p 8000:8000 ml-model:v1")
    print("   → Starts container from image")
    print()
    
    print("   3. Dockerfile (Recipe)")
    print("   ──────────────────────")
    print("   Think: Recipe card 📝")
    print()
    print("   • Text file with instructions")
    print("   • Each line = Layer")
    print("   • Cached for speed")
    print()
    print("   Basic structure:")
    print("   ```dockerfile")
    print("   FROM python:3.11-slim      # Base image")
    print("   WORKDIR /app               # Working directory")
    print("   COPY requirements.txt .    # Copy file")
    print("   RUN pip install -r requirements.txt  # Install deps")
    print("   COPY . .                   # Copy code")
    print("   CMD ['python', 'main.py']  # Default command")
    print("   ```")
    print()
    
    print("   4. Registry (Storage)")
    print("   ─────────────────────")
    print("   Think: App Store 🏪")
    print()
    print("   • Stores Docker images")
    print("   • Public: Docker Hub (hub.docker.com)")
    print("   • Private:")
    print("     - AWS ECR")
    print("     - GCP Container Registry")
    print("     - Azure Container Registry")
    print("     - Self-hosted")
    print()
    print("   Example:")
    print("   $ docker push myregistry/ml-model:v1")
    print("   → Uploads image to registry")
    print()
    
    print("🔧 ESSENTIAL DOCKER COMMANDS:")
    print()
    print("   Build Image:")
    print("   $ docker build -t ml-model:v1 .")
    print("     -t: Tag (name:version)")
    print("     .: Build context (current directory)")
    print()
    print("   Run Container:")
    print("   $ docker run -p 8000:8000 ml-model:v1")
    print("     -p: Port mapping (host:container)")
    print("     -d: Detached (background)")
    print("     -e: Environment variable")
    print("     -v: Volume mount (persistent data)")
    print()
    print("   List Containers:")
    print("   $ docker ps           # Running")
    print("   $ docker ps -a        # All")
    print()
    print("   Stop Container:")
    print("   $ docker stop <container_id>")
    print()
    print("   Remove Container:")
    print("   $ docker rm <container_id>")
    print()
    print("   List Images:")
    print("   $ docker images")
    print()
    print("   Remove Image:")
    print("   $ docker rmi ml-model:v1")
    print()
    print("   View Logs:")
    print("   $ docker logs <container_id>")
    print("     -f: Follow (tail)")
    print()
    print("   Execute Command:")
    print("   $ docker exec -it <container_id> bash")
    print("     -it: Interactive terminal")
    print()
    
    print("🏗️  DOCKERFILE BEST PRACTICES:")
    print()
    print("   1. Use specific base image:")
    print("   ❌ FROM python:3")
    print("   ✅ FROM python:3.11-slim")
    print()
    print("   Why: Reproducibility (exact version)")
    print()
    print("   2. Minimize layers:")
    print("   ❌ RUN apt-get update")
    print("      RUN apt-get install -y curl")
    print("      RUN apt-get install -y git")
    print()
    print("   ✅ RUN apt-get update && \\")
    print("       apt-get install -y curl git && \\")
    print("       rm -rf /var/lib/apt/lists/*")
    print()
    print("   Why: Smaller image, faster builds")
    print()
    print("   3. Order for cache efficiency:")
    print("   ✅ COPY requirements.txt .     # Changes rarely")
    print("      RUN pip install -r requirements.txt")
    print("      COPY . .                    # Changes often")
    print()
    print("   Why: Cache dependencies, rebuild only code")
    print()
    print("   4. Use .dockerignore:")
    print("   ```")
    print("   __pycache__")
    print("   *.pyc")
    print("   .git")
    print("   .venv")
    print("   tests/")
    print("   ```")
    print()
    print("   Why: Faster builds, smaller images")
    print()
    print("   5. Don't run as root:")
    print("   ✅ RUN useradd -m appuser")
    print("      USER appuser")
    print()
    print("   Why: Security (limit permissions)")


# ============================================================================
# 3. ML-Specific Dockerfile Patterns
# ============================================================================

def demo_ml_dockerfile_patterns():
    """
    Dockerfile patterns specific to ML models.
    """
    print("\n" + "=" * 70)
    print("3. ML-Specific Dockerfile Patterns")
    print("=" * 70)
    print()
    
    print("🎯 PATTERN 1: Simple FastAPI + Model")
    print()
    print("   Use case: Small model (<100MB), single file")
    print()
    print("   Dockerfile:")
    print("   ```dockerfile")
    print("   FROM python:3.11-slim")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Install dependencies")
    print("   COPY requirements.txt .")
    print("   RUN pip install --no-cache-dir -r requirements.txt")
    print("   ")
    print("   # Copy code")
    print("   COPY app/ ./app/")
    print("   ")
    print("   # Copy model")
    print("   COPY models/model.pkl ./models/")
    print("   ")
    print("   # Expose port")
    print("   EXPOSE 8000")
    print("   ")
    print("   # Run application")
    print("   CMD ['uvicorn', 'app.main:app', '--host', '0.0.0.0']")
    print("   ```")
    print()
    print("   Pros:")
    print("   ✅ Simple")
    print("   ✅ Fast builds")
    print("   ✅ Small image (~500MB)")
    print()
    print("   Cons:")
    print("   ❌ Model in image (large if model is big)")
    print()
    
    print("🎯 PATTERN 2: Multi-stage Build (Smaller Image)")
    print()
    print("   Use case: Reduce image size, remove build tools")
    print()
    print("   Dockerfile:")
    print("   ```dockerfile")
    print("   # Stage 1: Builder")
    print("   FROM python:3.11 as builder")
    print("   ")
    print("   WORKDIR /app")
    print("   COPY requirements.txt .")
    print("   RUN pip install --user -r requirements.txt")
    print("   ")
    print("   # Stage 2: Runtime")
    print("   FROM python:3.11-slim")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Copy only installed packages")
    print("   COPY --from=builder /root/.local /root/.local")
    print("   COPY app/ ./app/")
    print("   COPY models/ ./models/")
    print("   ")
    print("   ENV PATH=/root/.local/bin:$PATH")
    print("   ")
    print("   EXPOSE 8000")
    print("   CMD ['uvicorn', 'app.main:app', '--host', '0.0.0.0']")
    print("   ```")
    print()
    print("   Pros:")
    print("   ✅ Smaller final image (~300MB vs ~800MB)")
    print("   ✅ No build tools in production image")
    print("   ✅ More secure")
    print()
    print("   Cons:")
    print("   ⚠️  Slightly more complex")
    print()
    
    print("🎯 PATTERN 3: Download Model at Runtime")
    print()
    print("   Use case: Large model (>500MB), stored in S3/GCS")
    print()
    print("   Dockerfile:")
    print("   ```dockerfile")
    print("   FROM python:3.11-slim")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Install AWS CLI (for S3)")
    print("   RUN apt-get update && \\")
    print("       apt-get install -y awscli && \\")
    print("       rm -rf /var/lib/apt/lists/*")
    print("   ")
    print("   COPY requirements.txt .")
    print("   RUN pip install --no-cache-dir -r requirements.txt")
    print("   ")
    print("   COPY app/ ./app/")
    print("   COPY download_model.sh .")
    print("   ")
    print("   # Download model on startup")
    print("   ENTRYPOINT ['./download_model.sh']")
    print("   CMD ['uvicorn', 'app.main:app', '--host', '0.0.0.0']")
    print("   ```")
    print()
    print("   download_model.sh:")
    print("   ```bash")
    print("   #!/bin/bash")
    print("   aws s3 cp s3://my-bucket/model.pkl ./models/")
    print("   exec '$@'  # Run CMD")
    print("   ```")
    print()
    print("   Pros:")
    print("   ✅ Small image (no model inside)")
    print("   ✅ Easy model updates (just update S3)")
    print("   ✅ Multiple models without rebuilding")
    print()
    print("   Cons:")
    print("   ❌ Slower startup (download time)")
    print("   ❌ Requires internet/cloud storage")
    print()
    
    print("🎯 PATTERN 4: GPU Support (CUDA)")
    print()
    print("   Use case: PyTorch/TensorFlow models needing GPU")
    print()
    print("   Dockerfile:")
    print("   ```dockerfile")
    print("   # Use NVIDIA CUDA base image")
    print("   FROM nvidia/cuda:12.0.0-base-ubuntu22.04")
    print("   ")
    print("   # Install Python")
    print("   RUN apt-get update && \\")
    print("       apt-get install -y python3.11 python3-pip && \\")
    print("       rm -rf /var/lib/apt/lists/*")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Install PyTorch with CUDA support")
    print("   COPY requirements.txt .")
    print("   RUN pip install torch torchvision --index-url \\")
    print("       https://download.pytorch.org/whl/cu120")
    print("   RUN pip install -r requirements.txt")
    print("   ")
    print("   COPY app/ ./app/")
    print("   COPY models/ ./models/")
    print("   ")
    print("   EXPOSE 8000")
    print("   CMD ['uvicorn', 'app.main:app', '--host', '0.0.0.0']")
    print("   ```")
    print()
    print("   Run with GPU:")
    print("   ```bash")
    print("   docker run --gpus all -p 8000:8000 ml-model:v1")
    print("   ```")
    print()
    print("   Pros:")
    print("   ✅ GPU acceleration (10-100x faster)")
    print("   ✅ Supports deep learning models")
    print()
    print("   Cons:")
    print("   ❌ Large image (~2-5GB)")
    print("   ❌ Requires GPU hardware")
    print("   ❌ More expensive")
    print()
    
    print("📏 IMAGE SIZE COMPARISON:")
    print()
    print("   python:3.11:          1.0 GB")
    print("   python:3.11-slim:     0.2 GB  ✅")
    print("   python:3.11-alpine:   0.05 GB (⚠️  compatibility issues)")
    print("   nvidia/cuda:          2.5 GB")
    print()
    print("   Recommendation: python:3.11-slim (balance)")
    print()
    
    print("🔧 DOCKER COMPOSE FOR LOCAL DEV:")
    print()
    print("   docker-compose.yml:")
    print("   ```yaml")
    print("   version: '3.8'")
    print("   ")
    print("   services:")
    print("     api:")
    print("       build: .")
    print("       ports:")
    print("         - '8000:8000'")
    print("       environment:")
    print("         - MODEL_PATH=/app/models/model.pkl")
    print("       volumes:")
    print("         - ./models:/app/models  # Mount local models")
    print("         - ./app:/app/app        # Live code reload")
    print("     ")
    print("     redis:")
    print("       image: redis:7-alpine")
    print("       ports:")
    print("         - '6379:6379'")
    print("   ```")
    print()
    print("   Run:")
    print("   ```bash")
    print("   docker-compose up")
    print("   ```")
    print()
    print("   Benefits:")
    print("   ✅ Multi-service setup (API + Redis + DB)")
    print("   ✅ Easy local development")
    print("   ✅ Consistent team environment")


# ============================================================================
# 4. Production Deployment
# ============================================================================

def demo_production_deployment():
    """
    Deploy Docker containers to production.
    """
    print("\n" + "=" * 70)
    print("4. Production Deployment")
    print("=" * 70)
    print()
    
    print("☁️  DEPLOYMENT OPTIONS:")
    print()
    print("   1️⃣  AWS Fargate (Easiest)")
    print("   ────────────────────────")
    print("   What: Serverless containers")
    print("   No servers to manage")
    print()
    print("   Steps:")
    print("   1. Push image to ECR:")
    print("      ```bash")
    print("      aws ecr get-login-password | docker login ...")
    print("      docker tag ml-model:v1 123.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1")
    print("      docker push 123.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1")
    print("      ```")
    print()
    print("   2. Create ECS task definition")
    print("   3. Create ECS service")
    print("   4. Done! Auto-scaling included")
    print()
    print("   Pros:")
    print("   ✅ No servers to manage")
    print("   ✅ Auto-scaling")
    print("   ✅ Pay per use")
    print()
    print("   Cons:")
    print("   ❌ AWS-specific")
    print("   ❌ Cold starts (2-3 sec)")
    print()
    print("   Cost: ~$0.04/hr per task")
    print()
    
    print("   2️⃣  Google Cloud Run (Simplest)")
    print("   ────────────────────────────────")
    print("   What: Serverless containers")
    print("   Most developer-friendly")
    print()
    print("   Steps:")
    print("   1. Build & push:")
    print("      ```bash")
    print("      gcloud builds submit --tag gcr.io/my-project/ml-model:v1")
    print("      ```")
    print()
    print("   2. Deploy:")
    print("      ```bash")
    print("      gcloud run deploy ml-model \\")
    print("        --image gcr.io/my-project/ml-model:v1 \\")
    print("        --platform managed \\")
    print("        --region us-central1")
    print("      ```")
    print()
    print("   3. Done! Auto HTTPS, auto-scaling")
    print()
    print("   Pros:")
    print("   ✅ Simplest deployment")
    print("   ✅ Auto HTTPS")
    print("   ✅ Auto-scaling (0 to 1000)")
    print("   ✅ Free tier (2M requests/month)")
    print()
    print("   Cons:")
    print("   ❌ GCP-specific")
    print("   ❌ Request timeout (60 min max)")
    print()
    print("   Cost: $0.00002400/request")
    print()
    
    print("   3️⃣  Kubernetes (Most Powerful)")
    print("   ───────────────────────────────")
    print("   What: Container orchestration")
    print("   Industry standard for scale")
    print()
    print("   deployment.yaml:")
    print("   ```yaml")
    print("   apiVersion: apps/v1")
    print("   kind: Deployment")
    print("   metadata:")
    print("     name: ml-model")
    print("   spec:")
    print("     replicas: 3")
    print("     selector:")
    print("       matchLabels:")
    print("         app: ml-model")
    print("     template:")
    print("       metadata:")
    print("         labels:")
    print("           app: ml-model")
    print("       spec:")
    print("         containers:")
    print("         - name: ml-model")
    print("           image: myregistry/ml-model:v1")
    print("           ports:")
    print("           - containerPort: 8000")
    print("           resources:")
    print("             requests:")
    print("               cpu: 500m")
    print("               memory: 1Gi")
    print("             limits:")
    print("               cpu: 1000m")
    print("               memory: 2Gi")
    print("   ```")
    print()
    print("   Deploy:")
    print("   ```bash")
    print("   kubectl apply -f deployment.yaml")
    print("   ```")
    print()
    print("   Pros:")
    print("   ✅ Cloud-agnostic (AWS, GCP, Azure, on-prem)")
    print("   ✅ Advanced features (auto-scaling, self-healing)")
    print("   ✅ Industry standard")
    print("   ✅ Great for microservices")
    print()
    print("   Cons:")
    print("   ❌ Complex (steep learning curve)")
    print("   ❌ Requires cluster management")
    print()
    print("   Cost: ~$0.10/hr per node")
    print()
    
    print("🎯 WHICH TO CHOOSE?")
    print()
    print("   Starting out / Small team:")
    print("   → Google Cloud Run ✅")
    print("      (Simplest, generous free tier)")
    print()
    print("   AWS shop / Medium scale:")
    print("   → AWS Fargate ✅")
    print("      (Easy, integrates with AWS services)")
    print()
    print("   Large scale / Multi-cloud:")
    print("   → Kubernetes ✅")
    print("      (Most powerful, portable)")
    print()
    
    print("📊 MONITORING IN PRODUCTION:")
    print()
    print("   1. Health Checks:")
    print("   ```python")
    print("   @app.get('/health')")
    print("   async def health():")
    print("       return {'status': 'healthy', 'model_loaded': True}")
    print("   ```")
    print()
    print("   Configure in Kubernetes:")
    print("   ```yaml")
    print("   livenessProbe:")
    print("     httpGet:")
    print("       path: /health")
    print("       port: 8000")
    print("     initialDelaySeconds: 30")
    print("     periodSeconds: 10")
    print("   ```")
    print()
    print("   2. Metrics (Prometheus):")
    print("   ```python")
    print("   from prometheus_client import Counter, Histogram")
    print("   ")
    print("   prediction_count = Counter('predictions_total', 'Total predictions')")
    print("   latency = Histogram('prediction_latency_seconds', 'Prediction latency')")
    print("   ")
    print("   @app.post('/predict')")
    print("   async def predict(request):")
    print("       with latency.time():")
    print("           prediction = model.predict(request)")
    print("       prediction_count.inc()")
    print("       return prediction")
    print("   ```")
    print()
    print("   3. Logging:")
    print("   ```python")
    print("   import logging")
    print("   import json")
    print("   ")
    print("   # Structured logging (JSON)")
    print("   logging.info(json.dumps({")
    print("       'event': 'prediction',")
    print("       'user_id': request.user_id,")
    print("       'prediction': result,")
    print("       'latency_ms': latency")
    print("   }))")
    print("   ```")
    print()
    print("   Ship logs to:")
    print("   • CloudWatch (AWS)")
    print("   • Cloud Logging (GCP)")
    print("   • ELK Stack (Elasticsearch + Kibana)")
    print("   • Datadog")
    print()
    
    print("🚀 CI/CD PIPELINE:")
    print()
    print("   GitHub Actions example:")
    print("   ```yaml")
    print("   name: Deploy ML Model")
    print("   on:")
    print("     push:")
    print("       branches: [main]")
    print("   ")
    print("   jobs:")
    print("     deploy:")
    print("       runs-on: ubuntu-latest")
    print("       steps:")
    print("         - uses: actions/checkout@v3")
    print("         ")
    print("         - name: Build Docker image")
    print("           run: docker build -t ml-model:${{ github.sha }} .")
    print("         ")
    print("         - name: Push to registry")
    print("           run: |")
    print("             docker push myregistry/ml-model:${{ github.sha }}")
    print("         ")
    print("         - name: Deploy to Cloud Run")
    print("           run: |")
    print("             gcloud run deploy ml-model \\")
    print("               --image myregistry/ml-model:${{ github.sha }}")
    print("   ```")
    print()
    print("   Result: Push code → Auto-deploy! ✅")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🐳 Docker for ML Model Deployment\n")
    print("Learn how to containerize ML models for production!")
    print()
    
    demo_why_docker()
    demo_docker_fundamentals()
    demo_ml_dockerfile_patterns()
    demo_production_deployment()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Docker?
   - Reproducibility: Same environment everywhere
   - Isolation: No dependency conflicts
   - Portability: Run anywhere (laptop to cloud)
   - Scalability: 1 to 1000 containers
   - Fast deployment: docker run (done!)

2. Core Concepts:
   - Image: Blueprint (cookie cutter)
   - Container: Running instance (cookie)
   - Dockerfile: Recipe (how to build image)
   - Registry: Storage (Docker Hub, ECR, GCR)

3. Essential Commands:
   Build:  docker build -t ml-model:v1 .
   Run:    docker run -p 8000:8000 ml-model:v1
   Stop:   docker stop <container_id>
   Logs:   docker logs -f <container_id>
   Push:   docker push myregistry/ml-model:v1

4. Dockerfile Patterns:
   - Simple: Copy model into image
   - Multi-stage: Smaller final image
   - Download model: S3/GCS at runtime
   - GPU: CUDA base image

5. Production Deployment:
   Starting out → Google Cloud Run (easiest)
   AWS shop → AWS Fargate (managed)
   Large scale → Kubernetes (powerful)

Minimal Dockerfile for ML:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY app/ ./app/
COPY models/ ./models/

# Run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t ml-model:v1 .
docker run -p 8000:8000 ml-model:v1
```

Test:
```bash
curl http://localhost:8000/predict -X POST -H "Content-Type: application/json" -d '{"features": [...]}'
```

Production Checklist:
✅ Use specific base image (python:3.11-slim)
✅ Multi-stage build (smaller image)
✅ .dockerignore (faster builds)
✅ Don't run as root (security)
✅ Health check endpoint
✅ Structured logging (JSON)
✅ Prometheus metrics
✅ Environment variables (config)
✅ Volume mounts (persistent data)
✅ Resource limits (CPU, memory)

Resources:
- Docker docs: docs.docker.com
- Best practices: docs.docker.com/develop/dev-best-practices
- ML Docker examples: github.com/ml-tooling/ml-workspace

Congratulations! You now know how to:
• Containerize ML models with Docker
• Build efficient Docker images
• Deploy to cloud platforms
• Monitor production containers
• Set up CI/CD pipelines
""")


if __name__ == "__main__":
    main()
