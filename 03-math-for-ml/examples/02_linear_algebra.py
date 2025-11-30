"""
Linear Algebra for ML/AI Engineering

Build intuition for vectors, matrices, and operations you'll use in ML/AI.
Focus: Visual understanding and practical applications in neural networks.

Install: poetry add numpy matplotlib
Run: poetry run python 03-math-for-ml/examples/02_linear_algebra.py
"""

from typing import List, Tuple
import math


# ============================================================================
# 1. Vectors - Points in Space
# ============================================================================

def demo_vectors():
    """
    Vector: List of numbers representing a point or direction in space.
    
    INTUITION - Your Location on GPS:
    Imagine your position: [latitude, longitude]
    - Vector = [37.7749, -122.4194] is San Francisco
    - Each number is a coordinate (dimension)
    - More dimensions = more information about you
    
    Real ML Example:
    - User profile: [age=25, income=50k, credit_score=700]
    - Word "king": [0.234, -0.567, 0.891, ...] (1536 dimensions!)
    - Image pixel: [Red=255, Green=128, Blue=64]
    
    THE KEY INSIGHT:
    Everything in ML is a vector! Customer, word, image, song - all just lists of numbers.
    ML models? They're just functions that take vectors and output vectors.
    
    ML Intuition:
    - Data point: [age, income, credit_score]
    - Word embedding: 1536-dimensional vector
    - Image pixels: Vector of RGB values
    
    In ML, almost everything is a vector!
    """
    print("=" * 70)
    print("1. Vectors - Points in Space")
    print("=" * 70)
    print()
    print("💭 INTUITION: Your GPS Location")
    print("   Vector = [latitude, longitude]")
    print("   San Francisco = [37.7749, -122.4194]")
    print("   Each number = one piece of information (dimension)")
    print()
    print("   In ML: Everything becomes a vector!")
    print("   • You: [age, income, location, clicks, ...]")
    print("   • 'King': [0.234, -0.567, ...] 1536 dimensions!")
    print("   • Image pixel: [Red, Green, Blue]")
    print()
    
    # 2D vectors (easy to visualize)
    user_a = [25, 50000]  # [age, income]
    user_b = [30, 60000]
    user_c = [25, 55000]
    
    print("🧑 User vectors [age, income]:")
    print(f"  User A: {user_a}")
    print(f"  User B: {user_b}")
    print(f"  User C: {user_c}")
    print()
    
    # Vector length (magnitude)
    def vector_length(v: List[float]) -> float:
        """Calculate length of vector (distance from origin)."""
        return math.sqrt(sum(x**2 for x in v))
    
    print(f"Vector lengths (Euclidean distance from origin):")
    print(f"  |A| = {vector_length(user_a):.1f}")
    print(f"  |B| = {vector_length(user_b):.1f}")
    print()
    
    # Vector addition
    def add_vectors(v1: List[float], v2: List[float]) -> List[float]:
        """Add two vectors element-wise."""
        return [v1[i] + v2[i] for i in range(len(v1))]
    
    combined = add_vectors(user_a, [5, 10000])  # Age +5, income +10k
    print(f"User A after update: {combined}")
    print()
    
    # Scalar multiplication
    def scale_vector(v: List[float], scalar: float) -> List[float]:
        """Multiply vector by a scalar."""
        return [x * scalar for x in v]
    
    doubled = scale_vector(user_a, 2)
    print(f"User A doubled: {doubled}")
    print()
    
    print("💡 In ML:")
    print("   - Each data point is a vector")
    print("   - Each feature is a dimension")
    print("   - Model learns to position vectors in space")


# ============================================================================
# 2. Vector Operations - Similarity and Distance
# ============================================================================

def demo_vector_operations():
    """
    Key operations for ML:
    - Dot product: Measure similarity
    - Distance: Measure difference
    - Cosine similarity: Direction similarity
    
    INTUITION - Finding Compatible Friends:
    
    You and your friend rate interests (0-10):
    - You:    [Sports=8, Movies=9, Reading=3]
    - Friend: [Sports=7, Movies=8, Reading=2]
    
    Dot Product (similarity): 8*7 + 9*8 + 3*2 = 134 (high = similar tastes!)
    Distance: How far apart your preferences are
    Cosine Similarity: Do you like same things? (ignoring intensity)
    
    WHY IT MATTERS:
    - Dot product high → Recommend what your friend likes
    - Distance small → You'll enjoy same movies
    - Cosine similar → Similar personality, different intensity
    
    Real ML Uses:
    - Netflix: Which users are similar to you? (dot product)
    - Google Search: Which documents match your query? (cosine similarity)
    - Attention: Which words in sentence relate to each other? (dot product)
    
    ML Use Cases:
    - Recommendation systems (similar users/items)
    - Search (query vs documents)
    - Attention mechanisms (which words relate)
    """
    print("\n" + "=" * 70)
    print("2. Vector Operations - Similarity")
    print("=" * 70)
    print()
    print("💭 INTUITION: Finding Compatible Friends")
    print("   You rate interests [Sports, Movies, Reading]:")
    print("   You:    [8, 9, 3]")
    print("   Friend: [7, 8, 2]")
    print()
    print("   Dot Product: 8*7 + 9*8 + 3*2 = 134 (high = compatible!)")
    print("   → You both love movies & sports, meh on reading")
    print()
    print("   Distance: How different are your tastes?")
    print("   Cosine: Same taste direction? (intensity doesn't matter)")
    print()
    print("   ML: Netflix finds users like you, recommends their favorites!")
    print()
    
    # Word embeddings (simplified 3D for demo)
    word_king = [0.9, 0.8, 0.1]    # High on "royal", "male"
    word_queen = [0.9, 0.2, 0.9]   # High on "royal", "female"
    word_man = [0.1, 0.9, 0.1]     # High on "male"
    word_woman = [0.1, 0.1, 0.9]   # High on "female"
    
    print("🔤 Word Embeddings (simplified):")
    print(f"  king:  {word_king}")
    print(f"  queen: {word_queen}")
    print(f"  man:   {word_man}")
    print(f"  woman: {word_woman}")
    print()
    
    # Dot product (similarity measure)
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """
        Dot product: Sum of element-wise multiplication.
        
        Intuition:
        - High value → vectors point in similar direction
        - Zero → vectors are perpendicular
        - Negative → vectors point in opposite directions
        """
        return sum(v1[i] * v2[i] for i in range(len(v1)))
    
    print("Dot Products (similarity):")
    print(f"  king · queen = {dot_product(word_king, word_queen):.2f}")
    print(f"  king · man   = {dot_product(word_king, word_man):.2f}")
    print(f"  king · woman = {dot_product(word_king, word_woman):.2f}")
    print(f"  man · woman  = {dot_product(word_man, word_woman):.2f}")
    print()
    print("  → 'king' most similar to 'queen' (both royal)")
    print()
    
    # Euclidean distance
    def euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """Distance between two points."""
        return math.sqrt(sum((v1[i] - v2[i])**2 for i in range(len(v1))))
    
    print("Euclidean Distances (difference):")
    print(f"  king ↔ queen = {euclidean_distance(word_king, word_queen):.2f}")
    print(f"  king ↔ man   = {euclidean_distance(word_king, word_man):.2f}")
    print(f"  man ↔ woman  = {euclidean_distance(word_man, word_woman):.2f}")
    print()
    
    # Cosine similarity (direction, not magnitude)
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Cosine of angle between vectors.
        
        Range: -1 (opposite) to +1 (same direction)
        Good for: Text, embeddings (magnitude doesn't matter)
        """
        dot_prod = dot_product(v1, v2)
        mag_v1 = math.sqrt(sum(x**2 for x in v1))
        mag_v2 = math.sqrt(sum(x**2 for x in v2))
        return dot_prod / (mag_v1 * mag_v2)
    
    print("Cosine Similarity (direction):")
    print(f"  king ∠ queen = {cosine_similarity(word_king, word_queen):.3f}")
    print(f"  king ∠ man   = {cosine_similarity(word_king, word_man):.3f}")
    print(f"  man ∠ woman  = {cosine_similarity(word_man, word_woman):.3f}")
    print()
    
    print("💡 In ML:")
    print("   - Dot product: Attention scores, neural network layers")
    print("   - Distance: K-NN, clustering")
    print("   - Cosine similarity: Text similarity, recommendation")


# ============================================================================
# 3. Matrices - Transformations and Data
# ============================================================================

def demo_matrices():
    """
    Matrix: 2D array of numbers.
    
    INTUITION - Excel Spreadsheet:
    
    Think of a customer database in Excel:
    - Each ROW = One customer (person/sample)
    - Each COLUMN = One attribute (age/income/score)
    - Matrix = The whole spreadsheet!
    
    Example:
         Age | Income | Score
    A    25  |  50000 |   700
    B    30  |  60000 |   750
    C    35  |  55000 |   680
    
    Matrix = [[25, 50000, 700],
              [30, 60000, 750],
              [35, 55000, 680]]
    
    Shape: (3 rows, 3 columns) = (3, 3) = "3 by 3"
    
    WHY MATRICES IN ML?
    - Your entire dataset = one big matrix
    - Neural network weights = matrix that transforms data
    - Image = matrix where each cell is a pixel brightness
    - Video = sequence of image matrices
    
    THE MAGIC: Matrix multiplication = data transformation
    - Input matrix × Weight matrix = Output matrix
    - This is literally how neural networks work!
    
    ML Intuition:
    - Dataset: Each row is a sample, each column is a feature
    - Neural network layer: Transform input to output
    - Image: Matrix of pixel values
    
    Shape notation: (rows, columns) or (m, n)
    """
    print("\n" + "=" * 70)
    print("3. Matrices - Data and Transformations")
    print("=" * 70)
    print()
    print("💭 INTUITION: Excel Spreadsheet of Customers")
    print()
    print("        Age | Income | Score")
    print("   A    25  |  50000 |   700")
    print("   B    30  |  60000 |   750")
    print("   C    35  |  55000 |   680")
    print()
    print("   Each ROW = One customer (sample)")
    print("   Each COLUMN = One feature (age/income/score)")
    print("   Matrix = The whole spreadsheet!")
    print()
    print("   Your dataset IS a matrix.")
    print("   Neural networks? Just matrix multiplication!")
    print()
    
    # Dataset as matrix
    print("📊 Dataset as Matrix:")
    print()
    print("   User | Age | Income | Score")
    print("   --------------------------------")
    print("   A    |  25 |  50000 |   700")
    print("   B    |  30 |  60000 |   750")
    print("   C    |  35 |  55000 |   680")
    print()
    
    # Matrix representation
    data_matrix = [
        [25, 50000, 700],  # User A
        [30, 60000, 750],  # User B
        [35, 55000, 680],  # User C
    ]
    
    print(f"Matrix shape: {len(data_matrix)} rows × {len(data_matrix[0])} columns")
    print(f"Also written: (3, 3) or 3×3 matrix")
    print()
    
    # Transpose (swap rows and columns)
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Flip matrix over its diagonal."""
        rows = len(matrix)
        cols = len(matrix[0])
        return [[matrix[r][c] for r in range(rows)] for c in range(cols)]
    
    transposed = transpose(data_matrix)
    print("Transposed (features as rows):")
    print("  Age:    ", transposed[0])
    print("  Income: ", transposed[1])
    print("  Score:  ", transposed[2])
    print()
    
    # Neural network weight matrix
    print("🧠 Neural Network Weight Matrix:")
    print()
    print("   3 inputs → 2 outputs")
    print("   Weight matrix shape: (2, 3)")
    print()
    
    weights = [
        [0.5, 0.3, 0.2],  # Weights for output 1
        [0.4, 0.6, 0.1],  # Weights for output 2
    ]
    
    print("   Weights:", weights)
    print()
    print("   Each row: weights for one output neuron")
    print("   Each column: weights for one input feature")
    print()
    
    print("💡 In ML:")
    print("   - Data: Rows=samples, Columns=features")
    print("   - Weights: Each layer is a matrix")
    print("   - Operations: Matrix multiplication transforms data")


# ============================================================================
# 4. Matrix Multiplication - Neural Network Math
# ============================================================================

def demo_matrix_multiplication():
    """
    Matrix multiplication: The core operation in neural networks!
    
    Rule: (m×n) @ (n×p) = (m×p)
    
    INTUITION - Making Smoothies (Recipe Transformation):
    
    You have ingredients: [Banana=2, Strawberry=5, Yogurt=1]
    Recipe matrix transforms ingredients → smoothies:
    
                     Banana  Strawberry  Yogurt
    Tropical           3         2          1      (needs 3 bananas, 2 strawberries, 1 yogurt)
    Berry Blast        1         5          2      (needs 1 banana, 5 strawberries, 2 yogurt)
    
    Your ingredients × Recipe = How many smoothies you can make!
    
    [2, 5, 1] × [[3, 2, 1],    →  [Tropical=?, Berry Blast=?]
                 [1, 5, 2]]
    
    This is EXACTLY what neural networks do:
    - Input = Your data [age, income, clicks]
    - Weights = Recipe (learned from training)
    - Output = Predictions [will_buy, will_churn]
    
    THE KEY: Matrix multiplication = transformation
    Every layer in a neural network is one matrix multiplication!
    
    ML Use Case:
    - Every neural network layer is matrix multiplication
    - Transforms input vector to output vector
    """
    print("\n" + "=" * 70)
    print("4. Matrix Multiplication - Neural Network Core")
    print("=" * 70)
    print()
    print("💭 INTUITION: Making Smoothies from Ingredients")
    print()
    print("   Ingredients: [Banana=2, Strawberry=5, Yogurt=1]")
    print()
    print("   Recipe Matrix:")
    print("                  Banana  Straw  Yogurt")
    print("   Tropical         3      2      1")
    print("   Berry Blast      1      5      2")
    print()
    print("   Ingredients × Recipe = Smoothies you can make!")
    print()
    print("   Neural Network = Same thing!")
    print("   Input × Weights = Output predictions")
    print("   This is literally how every layer works!")
    print()
    
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """
        Multiply two matrices.
        
        Formula: C[i][j] = sum(A[i][k] * B[k][j] for k)
        
        Intuition: Each output is dot product of row from A, column from B
        """
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    # Example: Simple neural network layer
    print("🧠 Neural Network Layer:")
    print()
    print("   Input: 3 features [age, income, score]")
    print("   Output: 2 neurons")
    print()
    
    # Input vector (as 1×3 matrix)
    input_vector = [[25, 50000, 700]]
    
    # Weight matrix (2×3)
    weights = [
        [0.001, 0.00001, 0.002],  # Neuron 1 weights
        [0.002, 0.00002, 0.001],  # Neuron 2 weights
    ]
    
    print(f"   Input (1×3): {input_vector[0]}")
    print(f"   Weights (2×3):")
    for i, w in enumerate(weights):
        print(f"      Neuron {i+1}: {w}")
    print()
    
    # Matrix multiplication: (1×3) @ (3×2) = (1×2)
    # Need to transpose weights for standard multiplication
    weights_T = [[weights[j][i] for j in range(len(weights))] for i in range(len(weights[0]))]
    output = matrix_multiply(input_vector, weights_T)
    
    print(f"   Output (1×2): {output[0]}")
    print()
    print("   Calculation for Neuron 1:")
    print(f"      = {input_vector[0][0]}*{weights[0][0]} + "
          f"{input_vector[0][1]}*{weights[0][1]} + "
          f"{input_vector[0][2]}*{weights[0][2]}")
    print(f"      = {output[0][0]:.2f}")
    print()
    
    # Batch processing
    print("📦 Batch Processing (Multiple Inputs):")
    print()
    
    batch_inputs = [
        [25, 50000, 700],  # User A
        [30, 60000, 750],  # User B
        [35, 55000, 680],  # User C
    ]
    
    print(f"   Batch (3×3):")
    for i, inp in enumerate(batch_inputs):
        print(f"      User {chr(65+i)}: {inp}")
    print()
    
    # Batch matrix multiplication: (3×3) @ (3×2) = (3×2)
    batch_output = matrix_multiply(batch_inputs, weights_T)
    
    print(f"   Output (3×2):")
    for i, out in enumerate(batch_output):
        print(f"      User {chr(65+i)}: {out}")
    print()
    
    print("💡 In ML:")
    print("   - Single input: (1×n) @ (n×m) = (1×m)")
    print("   - Batch: (batch_size×n) @ (n×m) = (batch_size×m)")
    print("   - Each layer: Matrix multiplication + activation")


# ============================================================================
# 5. Eigenvalues and Eigenvectors - PCA
# ============================================================================

def demo_eigenvectors():
    """
    Eigenvector: Special vector that doesn't change direction when transformed.
    Eigenvalue: How much the eigenvector is scaled.
    
    Formula: A @ v = λ @ v
    - A: Matrix
    - v: Eigenvector
    - λ: Eigenvalue
    
    ML Use Case:
    - PCA (Principal Component Analysis)
    - Dimensionality reduction
    - Finding important directions in data
    """
    print("\n" + "=" * 70)
    print("5. Eigenvalues/Eigenvectors - Finding Important Directions")
    print("=" * 70)
    
    print("🎯 Intuition: Eigenvectors are 'special directions'")
    print()
    print("   When you transform data with a matrix,")
    print("   most vectors change direction.")
    print("   Eigenvectors only get stretched/shrunk!")
    print()
    
    # Simple 2×2 example
    print("Example Matrix:")
    A = [[2, 1],
         [1, 2]]
    
    for row in A:
        print(f"   {row}")
    print()
    
    print("Eigenvector 1: [1, 1]")
    v1 = [1, 1]
    print(f"   A @ v1 = [{A[0][0]*v1[0] + A[0][1]*v1[1]}, "
          f"{A[1][0]*v1[0] + A[1][1]*v1[1]}]")
    print(f"         = [3, 3]")
    print(f"         = 3 * [1, 1]  ← Scaled by 3!")
    print(f"   Eigenvalue λ1 = 3")
    print()
    
    print("Eigenvector 2: [1, -1]")
    v2 = [1, -1]
    print(f"   A @ v2 = [{A[0][0]*v2[0] + A[0][1]*v2[1]}, "
          f"{A[1][0]*v2[0] + A[1][1]*v2[1]}]")
    print(f"         = [1, -1]")
    print(f"         = 1 * [1, -1]  ← Scaled by 1!")
    print(f"   Eigenvalue λ2 = 1")
    print()
    
    # PCA example (conceptual)
    print("📊 ML Application: PCA (Dimensionality Reduction)")
    print()
    print("   Problem: 1000 features, too many!")
    print("   Solution: Find most important directions (eigenvectors)")
    print()
    print("   Steps:")
    print("   1. Center data (subtract mean)")
    print("   2. Compute covariance matrix")
    print("   3. Find eigenvectors of covariance matrix")
    print("   4. Keep top eigenvectors (highest eigenvalues)")
    print("   5. Project data onto these directions")
    print()
    print("   Result: 1000 features → 50 features")
    print("           Keeping 95% of information!")
    print()
    
    # Simulate PCA
    print("Example: 3D → 2D reduction")
    print()
    
    data_3d = [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 7],
        [4, 7, 9],
    ]
    
    print("   Original data (4 samples × 3 features):")
    for i, point in enumerate(data_3d):
        print(f"      Sample {i+1}: {point}")
    print()
    
    # In real PCA, we'd:
    # 1. Center data
    # 2. Compute covariance
    # 3. Find eigenvectors
    # 4. Project onto top 2 eigenvectors
    
    # Simulated result (after PCA)
    data_2d = [
        [2.1, 0.1],
        [4.3, 0.2],
        [6.2, -0.1],
        [8.5, 0.0],
    ]
    
    print("   After PCA (4 samples × 2 features):")
    for i, point in enumerate(data_2d):
        print(f"      Sample {i+1}: {point}")
    print()
    
    print("   Benefit: Faster training, less overfitting!")
    print()
    
    print("💡 In ML:")
    print("   - PCA: Dimensionality reduction")
    print("   - Largest eigenvalue: Most important direction")
    print("   - Eigenvectors: Principal components")


# ============================================================================
# 6. Real-World: Similarity Search
# ============================================================================

def demo_similarity_search():
    """
    Real-world ML: Find similar items (users, products, documents).
    
    Use Cases:
    - Recommendation systems
    - Search engines
    - Duplicate detection
    - Vector databases (ChromaDB, Pinecone)
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Similarity Search")
    print("=" * 70)
    
    print("🔍 Scenario: Movie Recommendation")
    print()
    
    # Movie embeddings (features: action, comedy, drama, romance)
    movies = {
        "The Matrix": [0.9, 0.1, 0.5, 0.1],
        "Die Hard": [0.95, 0.3, 0.2, 0.1],
        "The Notebook": [0.1, 0.2, 0.8, 0.95],
        "Superbad": [0.2, 0.95, 0.3, 0.4],
        "Inception": [0.85, 0.1, 0.7, 0.2],
        "Bridesmaids": [0.1, 0.9, 0.4, 0.5],
    }
    
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_prod = sum(v1[i] * v2[i] for i in range(len(v1)))
        mag_v1 = math.sqrt(sum(x**2 for x in v1))
        mag_v2 = math.sqrt(sum(x**2 for x in v2))
        return dot_prod / (mag_v1 * mag_v2)
    
    def find_similar(query_movie: str, top_k: int = 3):
        """Find top-k similar movies."""
        query_vec = movies[query_movie]
        similarities = []
        
        for title, vec in movies.items():
            if title != query_movie:
                sim = cosine_similarity(query_vec, vec)
                similarities.append((title, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # Find movies similar to "The Matrix"
    query = "The Matrix"
    print(f"User watched: '{query}'")
    print(f"Features: {movies[query]}")
    print(f"          [action, comedy, drama, romance]")
    print()
    print("Similar movies:")
    
    similar = find_similar(query, top_k=3)
    for i, (title, sim) in enumerate(similar, 1):
        print(f"  {i}. {title}")
        print(f"     Similarity: {sim:.3f}")
        print(f"     Features: {movies[title]}")
    print()
    
    # Another example
    query2 = "Superbad"
    print(f"\nUser watched: '{query2}'")
    print("Similar movies:")
    
    similar2 = find_similar(query2, top_k=2)
    for i, (title, sim) in enumerate(similar2, 1):
        print(f"  {i}. {title} (similarity: {sim:.3f})")
    print()
    
    print("💡 This is how:")
    print("   - Netflix recommends movies")
    print("   - Spotify finds similar songs")
    print("   - Amazon suggests products")
    print("   - Vector databases work (RAG systems)")


# ============================================================================
# 7. Real-World: Attention Mechanism
# ============================================================================

def demo_attention():
    """
    Real-world ML: Attention mechanism in transformers (GPT, BERT).
    
    Key operation: Scaled dot-product attention
    - Query: What I'm looking for
    - Key: What others offer
    - Value: The actual content
    
    Attention score: How much to focus on each element
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Attention Mechanism (Transformers)")
    print("=" * 70)
    
    print("🤖 Simplified Attention: 'Which words relate to each other?'")
    print()
    
    # Sentence: "The cat sat on the mat"
    words = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Simplified embeddings (2D for demo)
    embeddings = {
        "The": [0.1, 0.2],
        "cat": [0.8, 0.3],
        "sat": [0.3, 0.7],
        "on": [0.2, 0.1],
        "the": [0.1, 0.2],
        "mat": [0.7, 0.4],
    }
    
    print(f"Sentence: {' '.join(words)}")
    print()
    
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """Calculate dot product."""
        return sum(v1[i] * v2[i] for i in range(len(v1)))
    
    def softmax(scores: List[float]) -> List[float]:
        """Convert scores to probabilities (sum to 1)."""
        # Simplified softmax
        exp_scores = [math.exp(s) for s in scores]
        total = sum(exp_scores)
        return [s / total for s in exp_scores]
    
    # Calculate attention for word "cat"
    query_word = "cat"
    query_vec = embeddings[query_word]
    
    print(f"Query word: '{query_word}'")
    print(f"Query vector: {query_vec}")
    print()
    
    # Calculate attention scores (how much cat relates to each word)
    print("Attention scores (dot products):")
    scores = []
    for word in words:
        key_vec = embeddings[word]
        score = dot_product(query_vec, key_vec)
        scores.append(score)
        print(f"  '{query_word}' · '{word}' = {score:.3f}")
    print()
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    print("Attention weights (after softmax):")
    for i, word in enumerate(words):
        bar = "█" * int(attention_weights[i] * 50)
        print(f"  '{word}': {bar} {attention_weights[i]:.3f}")
    print()
    
    print(f"Sum of weights: {sum(attention_weights):.3f} (should be 1.0)")
    print()
    
    print("Interpretation:")
    print(f"  When processing 'cat', the model pays attention to:")
    max_idx = attention_weights.index(max(attention_weights))
    print(f"  - Mostly: '{words[max_idx]}' ({attention_weights[max_idx]:.1%})")
    print(f"  - Also: 'mat', 'sat' (nouns and verbs relate)")
    print(f"  - Less: 'the', 'on' (function words)")
    print()
    
    print("💡 In transformers (GPT, BERT):")
    print("   - Every word attends to every other word")
    print("   - Multiple attention heads learn different patterns")
    print("   - This is why transformers understand context!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🔢 Linear Algebra for ML/AI Engineering\n")
    print("Focus: Build intuition for neural network math!")
    print()
    
    demo_vectors()
    demo_vector_operations()
    demo_matrices()
    demo_matrix_multiplication()
    demo_eigenvectors()
    demo_similarity_search()
    demo_attention()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Vectors: Data points in space
   - Each feature is a dimension
   - Operations: add, scale, dot product

2. Dot Product: Similarity measure
   - High → similar direction
   - Zero → perpendicular
   - Core of neural networks!

3. Matrices: Data and transformations
   - Rows = samples, Columns = features
   - Each NN layer = matrix

4. Matrix Multiplication: NN forward pass
   - (batch×input) @ (input×output) = (batch×output)
   - Every layer does this!

5. Eigenvalues/Eigenvectors: Important directions
   - PCA: Dimensionality reduction
   - Find patterns in data

6. Distance Metrics:
   - Euclidean: Spatial distance
   - Cosine: Directional similarity

ML Applications:
- Dot product: Attention, neural layers
- Matrix mult: Every NN layer
- Cosine similarity: Recommendations, search
- PCA: Reduce features
- Distance: K-NN, clustering

Next: 03_calculus_essentials.py - Derivatives, gradient descent!
""")


if __name__ == "__main__":
    main()
