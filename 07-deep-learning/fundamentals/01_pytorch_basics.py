"""
PyTorch Basics for Deep Learning

Learn PyTorch fundamentals: tensors, autograd, neural networks from scratch.
Focus: Understanding HOW PyTorch works and WHY it's powerful for deep learning.

Install: poetry add torch torchvision
Run: poetry run python 07-deep-learning/fundamentals/01_pytorch_basics.py
"""

import random
import math
from typing import List, Tuple


# ============================================================================
# 1. Tensors - The Foundation
# ============================================================================

def demo_tensors():
    """
    Tensors: Multi-dimensional arrays (PyTorch's core data structure)
    
    INTUITION - Think of Data Dimensions:
    
    Scalar (0D): Single number
    - Example: Temperature = 72°F
    - Use: Loss value, accuracy score
    
    Vector (1D): List of numbers
    - Example: Daily temperatures [70, 72, 68, 75, 73]
    - Use: Word embedding, single image row
    
    Matrix (2D): Table of numbers
    - Example: Grayscale image (28x28 pixels)
    - Use: Spreadsheet, grayscale image, feature matrix
    
    3D Tensor: Stack of matrices
    - Example: RGB image (height × width × 3 channels)
    - Use: Color image, video frame, batch of sequences
    
    4D Tensor: Batch of 3D tensors
    - Example: Batch of 32 RGB images (32 × 3 × 224 × 224)
    - Use: Training batch of images
    
    WHY TENSORS?
    - GPU acceleration (100x faster than CPU)
    - Automatic differentiation (backpropagation for free!)
    - Efficient batch operations (process 1000s of examples at once)
    
    Real Impact:
    Without tensors: Train model in 10 hours on CPU
    With tensors: Train same model in 6 minutes on GPU!
    """
    print("=" * 70)
    print("1. Tensors - Multi-dimensional Arrays")
    print("=" * 70)
    print()
    print("💭 INTUITION: Dimensions of Data")
    print()
    print("   0D (Scalar): Single number")
    print("   Example: loss = 0.42")
    print()
    print("   1D (Vector): List of numbers")
    print("   Example: temperatures = [70, 72, 68, 75]")
    print()
    print("   2D (Matrix): Table of numbers")
    print("   Example: Grayscale image (28×28 pixels)")
    print()
    print("   3D (Tensor): Stack of matrices")
    print("   Example: RGB image (height × width × 3 colors)")
    print()
    print("   4D (Batch): Multiple 3D tensors")
    print("   Example: Batch of 32 images (32 × 3 × 224 × 224)")
    print()
    
    # Create tensors (simulated - we'll show the concept)
    print("📊 Creating Tensors (Conceptual):")
    print()
    
    # Scalar
    scalar = 42
    print(f"Scalar: {scalar}")
    print(f"  Shape: () - just a single number")
    print(f"  Use: Loss value, accuracy metric")
    print()
    
    # Vector
    vector = [1, 2, 3, 4, 5]
    print(f"Vector: {vector}")
    print(f"  Shape: (5,) - 5 elements")
    print(f"  Use: Word embedding, feature vector")
    print()
    
    # Matrix
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print(f"Matrix:")
    for row in matrix:
        print(f"  {row}")
    print(f"  Shape: (3, 3) - 3 rows × 3 columns")
    print(f"  Use: Grayscale image, feature matrix")
    print()
    
    # 3D Tensor (RGB image representation)
    print("3D Tensor (RGB Image - 2×2×3):")
    rgb_image = [
        [[255, 0, 0], [0, 255, 0]],      # Row 1: Red, Green
        [[0, 0, 255], [255, 255, 255]]   # Row 2: Blue, White
    ]
    print("  Channel view (R, G, B):")
    print(f"    Red channel:   [[255, 0], [0, 255]]")
    print(f"    Green channel: [[0, 255], [0, 255]]")
    print(f"    Blue channel:  [[0, 0], [255, 255]]")
    print(f"  Shape: (2, 2, 3) - height × width × channels")
    print()
    
    # 4D Tensor (batch)
    print("4D Tensor (Batch of Images):")
    print("  Shape: (32, 3, 224, 224)")
    print("  Meaning:")
    print("    32  = batch size (32 images)")
    print("    3   = channels (RGB)")
    print("    224 = height in pixels")
    print("    224 = width in pixels")
    print("  Total elements: 32 × 3 × 224 × 224 = 4,816,896 numbers!")
    print()
    
    print("💡 Real PyTorch Code (for reference):")
    print("""
    import torch
    
    # Scalar
    scalar = torch.tensor(42)
    
    # Vector
    vector = torch.tensor([1, 2, 3, 4, 5])
    
    # Matrix
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    # Random 4D tensor (batch of images)
    batch = torch.randn(32, 3, 224, 224)
    
    # Move to GPU for 100x speedup
    if torch.cuda.is_available():
        batch = batch.cuda()
    """)
    print()
    
    print("🚀 WHY TENSORS ARE POWERFUL:")
    print()
    print("   1️⃣  GPU Acceleration:")
    print("      CPU: Process one number at a time")
    print("      GPU: Process millions of numbers simultaneously!")
    print("      Result: 100-1000x speedup")
    print()
    print("   2️⃣  Batch Operations:")
    print("      Old way: Loop through each image (slow)")
    print("      Tensor way: Process entire batch at once (fast)")
    print("      Training 1000 images: 10 seconds vs 0.1 seconds")
    print()
    print("   3️⃣  Automatic Differentiation:")
    print("      You: Define forward pass")
    print("      PyTorch: Calculates gradients automatically!")
    print("      No manual backprop math needed")


# ============================================================================
# 2. Automatic Differentiation (Autograd)
# ============================================================================

def demo_autograd():
    """
    Autograd: Automatic gradient computation for backpropagation
    
    INTUITION - The Hiking Analogy:
    
    You're lost on a mountain in fog, trying to reach the valley (minimum).
    
    Without gradients:
    - Walk random directions, hope to descend
    - Might go uphill, waste time, never find bottom
    
    With gradients:
    - Check slope under your feet
    - Walk in steepest downhill direction
    - Efficiently reach the valley!
    
    In ML:
    - Mountain = Loss function (error)
    - Valley = Best model parameters
    - Gradient = Which direction to adjust weights
    
    WHY AUTOGRAD IS MAGICAL:
    
    Old Days (1990s):
    1. Write forward pass (model prediction)
    2. Manually derive backprop equations (calculus nightmare!)
    3. Implement backward pass (error-prone)
    4. Debug for weeks
    
    With PyTorch:
    1. Write forward pass
    2. Call .backward()
    3. Done! Gradients computed automatically ✨
    
    Real Impact:
    - PhD-level math → Not needed anymore!
    - New model architectures → Easy to try
    - Complex networks → Same simple code
    
    Example:
    Loss = (prediction - actual)²
    
    Manual: d(Loss)/d(weight) = 2 × (pred - actual) × d(pred)/d(weight)
    PyTorch: loss.backward()  # All gradients computed!
    """
    print("\n" + "=" * 70)
    print("2. Automatic Differentiation (Autograd)")
    print("=" * 70)
    print()
    print("💭 INTUITION: Lost on a Mountain in Fog")
    print()
    print("   Goal: Reach valley (lowest point)")
    print()
    print("   ❌ Without Gradients:")
    print("      Walk random directions → Might go uphill!")
    print("      Slow, inefficient, might never find bottom")
    print()
    print("   ✅ With Gradients:")
    print("      Check slope → Walk downhill")
    print("      Fast, efficient, guaranteed to descend!")
    print()
    print("   In ML:")
    print("   • Mountain = Loss (error)")
    print("   • Valley = Best weights")
    print("   • Gradient = Direction to improve")
    print()
    
    print("🎯 Simple Example: f(x) = x²")
    print()
    
    # Simulate gradient calculation
    x = 3.0
    y = x ** 2  # y = 9
    
    # Derivative: dy/dx = 2x
    gradient = 2 * x  # = 6
    
    print(f"Function: f(x) = x²")
    print(f"At x = {x}:")
    print(f"  f(x) = {y}")
    print(f"  Gradient df/dx = {gradient}")
    print()
    print("💡 Gradient tells us: 'If x increases by 1, f increases by ~6'")
    print()
    
    print("🧠 Neural Network Example:")
    print()
    print("  Imagine: weight = 0.5, input = 2.0, target = 3.0")
    print()
    
    # Forward pass
    weight = 0.5
    x_input = 2.0
    prediction = weight * x_input  # = 1.0
    target = 3.0
    loss = (prediction - target) ** 2  # = 4.0
    
    print(f"  Forward Pass:")
    print(f"    prediction = weight × input = {weight} × {x_input} = {prediction}")
    print(f"    loss = (prediction - target)² = ({prediction} - {target})² = {loss}")
    print()
    
    # Backward pass (manual for demonstration)
    # dL/dw = dL/dp × dp/dw
    # dL/dp = 2 × (prediction - target)
    # dp/dw = x_input
    
    dL_dp = 2 * (prediction - target)  # = -4
    dp_dw = x_input  # = 2.0
    dL_dw = dL_dp * dp_dw  # = -8
    
    print(f"  Backward Pass (Gradient Calculation):")
    print(f"    dL/d(prediction) = 2 × (pred - target) = {dL_dp}")
    print(f"    d(prediction)/d(weight) = input = {dp_dw}")
    print(f"    dL/d(weight) = {dL_dp} × {dp_dw} = {dL_dw}")
    print()
    print(f"  💡 Gradient = {dL_dw} means:")
    print(f"     'Loss decreased by 8 per unit increase in weight'")
    print(f"     'Weight is too small, increase it!'")
    print()
    
    # Update weight
    learning_rate = 0.1
    new_weight = weight - learning_rate * dL_dw
    
    print(f"  Weight Update:")
    print(f"    new_weight = old_weight - lr × gradient")
    print(f"    new_weight = {weight} - {learning_rate} × {dL_dw}")
    print(f"    new_weight = {new_weight}")
    print()
    
    # Check new loss
    new_prediction = new_weight * x_input
    new_loss = (new_prediction - target) ** 2
    
    print(f"  After Update:")
    print(f"    new prediction = {new_prediction}")
    print(f"    new loss = {new_loss:.4f}")
    print(f"    Improvement: {loss:.4f} → {new_loss:.4f} ✅")
    print()
    
    print("🎉 THE MAGIC: PyTorch Does This Automatically!")
    print()
    print("  Manual (what we just did):")
    print("    • Calculate forward pass")
    print("    • Derive gradient equations (calculus)")
    print("    • Implement backward pass")
    print("    • Debug gradient bugs")
    print("    Time: Hours to days")
    print()
    print("  PyTorch (what you actually write):")
    print("""
    # Forward pass
    prediction = weight * x
    loss = (prediction - target) ** 2
    
    # Backward pass (all gradients computed!)
    loss.backward()
    
    # Gradients ready to use
    gradient = weight.grad
    """)
    print("    Time: 3 lines of code!")
    print()
    
    print("💡 Real-World Impact:")
    print()
    print("   Before Autograd (1990s-2000s):")
    print("   • Simple networks only (manual math too hard)")
    print("   • Months to implement new architecture")
    print("   • PhD required to understand backprop")
    print()
    print("   With Autograd (2010s+):")
    print("   • Any architecture you can imagine")
    print("   • Try new ideas in minutes")
    print("   • Focus on creativity, not calculus!")
    print()
    print("   This is why deep learning exploded in 2010s!")


# ============================================================================
# 3. Building a Neural Network from Scratch
# ============================================================================

class SimpleNeuralNetwork:
    """
    Simple 2-layer neural network (from scratch for learning)
    
    Architecture:
    Input → Hidden Layer (ReLU) → Output Layer
    
    INTUITION - The Team Decision Analogy:
    
    You manage a team deciding whether to launch a product.
    
    Layer 1 (Hidden): Specialists analyze different aspects
    - Neuron 1: Market specialist (looks at customer data)
    - Neuron 2: Finance specialist (looks at cost data)
    - Neuron 3: Tech specialist (looks at feasibility)
    
    Each specialist:
    1. Gets all input data
    2. Weights what matters to them (learned weights)
    3. Makes a judgment (activation)
    
    Layer 2 (Output): CEO makes final decision
    - Combines all specialist opinions
    - Weighted by trust (learned weights)
    - Outputs final yes/no (prediction)
    
    Training = Learning which weights to trust!
    
    Why Hidden Layers?
    - Extract features: First layer finds edges, textures
    - Combine features: Next layer finds shapes, patterns
    - Make decision: Output layer classifies
    
    Example: Image of cat
    Layer 1: Detects edges, colors, textures
    Layer 2: Combines into ears, whiskers, fur patterns
    Output: "It's a cat!" (98% confidence)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
        """
        # Initialize weights randomly (small values)
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] 
                   for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] 
                   for _ in range(output_size)]
        self.b2 = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
        
        # Store for backprop
        self.cache = {}
    
    def relu(self, x: List[float]) -> List[float]:
        """ReLU activation: max(0, x)"""
        return [max(0, val) for val in x]
    
    def relu_derivative(self, x: List[float]) -> List[float]:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return [1.0 if val > 0 else 0.0 for val in x]
    
    def sigmoid(self, x: List[float]) -> List[float]:
        """Sigmoid activation: 1 / (1 + e^-x)"""
        return [1 / (1 + math.exp(-val)) for val in x]
    
    def matrix_multiply(self, A: List[List[float]], x: List[float]) -> List[float]:
        """Multiply matrix A by vector x"""
        result = []
        for row in A:
            result.append(sum(a * b for a, b in zip(row, x)))
        return result
    
    def vector_add(self, a: List[float], b: List[float]) -> List[float]:
        """Add two vectors"""
        return [x + y for x, y in zip(a, b)]
    
    def forward(self, X: List[float]) -> List[float]:
        """
        Forward pass through network.
        
        Args:
            X: Input features
        
        Returns:
            predictions: Output predictions
        """
        # Hidden layer: z1 = W1 × X + b1
        z1 = self.vector_add(self.matrix_multiply(self.W1, X), self.b1)
        a1 = self.relu(z1)  # Activation
        
        # Output layer: z2 = W2 × a1 + b2
        z2 = self.vector_add(self.matrix_multiply(self.W2, a1), self.b2)
        a2 = self.sigmoid(z2)  # Final activation
        
        # Cache for backprop
        self.cache = {
            'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2
        }
        
        return a2
    
    def compute_loss(self, predictions: List[float], targets: List[float]) -> float:
        """Mean Squared Error loss"""
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def demo_neural_network():
    """
    Build and train a simple neural network from scratch.
    
    Problem: Binary classification (is it a cat?)
    Input: 2 features (simplified)
    Output: 1 value (probability it's a cat)
    """
    print("\n" + "=" * 70)
    print("3. Neural Network from Scratch")
    print("=" * 70)
    print()
    print("💭 INTUITION: Team Decision Making")
    print()
    print("   Problem: Should we launch this product?")
    print()
    print("   Layer 1 (Hidden): Specialists analyze")
    print("   • Neuron 1: Market specialist")
    print("   • Neuron 2: Finance specialist")
    print("   • Neuron 3: Tech specialist")
    print()
    print("   Each specialist:")
    print("   1. Reviews all data")
    print("   2. Weights what matters to them")
    print("   3. Gives opinion (activation)")
    print()
    print("   Layer 2 (Output): CEO decides")
    print("   • Combines specialist opinions")
    print("   • Weights by trust (learned)")
    print("   • Final decision: yes/no")
    print()
    print("   Training = Learning which opinions to trust!")
    print()
    
    print("🎯 Our Problem: Classify Points (Simple XOR)")
    print()
    print("   Data points:")
    print("   • (0, 0) → 0  (bottom-left)")
    print("   • (0, 1) → 1  (top-left)")
    print("   • (1, 0) → 1  (bottom-right)")
    print("   • (1, 1) → 0  (top-right)")
    print()
    print("   Pattern: XOR (exclusive or)")
    print("   Can't solve with single line (need hidden layer!)")
    print()
    
    # Create network
    network = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Training data (XOR problem)
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y_train = [[0], [1], [1], [0]]
    
    print("📊 Network Architecture:")
    print(f"  Input layer: 2 neurons (x1, x2)")
    print(f"  Hidden layer: 4 neurons (ReLU activation)")
    print(f"  Output layer: 1 neuron (Sigmoid activation)")
    print(f"  Total parameters: {2*4 + 4 + 4*1 + 1} = 17")
    print()
    
    # Test before training
    print("🔍 Before Training:")
    for X, y in zip(X_train, y_train):
        pred = network.forward(X)
        print(f"  Input {X} → Prediction {pred[0]:.3f}, Target {y[0]}")
    print("  (Random predictions, not useful yet)")
    print()
    
    print("🏋️ Training Simulation:")
    print("  (In real PyTorch, we'd use optimizers & autograd)")
    print("  Here: Showing the concept")
    print()
    
    # Simple training loop simulation
    learning_rate = 0.1
    epochs = 5
    
    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            # Forward pass
            pred = network.forward(X)
            loss = network.compute_loss(pred, y)
            total_loss += loss
            
            # (In real training, backward pass would update weights here)
        
        avg_loss = total_loss / len(X_train)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    print()
    print("💡 What Happened During Training:")
    print()
    print("   Each iteration:")
    print("   1. Forward pass: Make prediction")
    print("   2. Compute loss: How wrong were we?")
    print("   3. Backward pass: Calculate gradients")
    print("   4. Update weights: Adjust to reduce loss")
    print()
    print("   After many iterations:")
    print("   • Hidden neurons learn to detect patterns")
    print("   • Output neuron learns to combine them")
    print("   • Network can solve XOR problem!")
    print()
    
    print("🎯 Real PyTorch Code (for reference):")
    print("""
    import torch
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.output = nn.Linear(4, 1)
            
        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = torch.sigmoid(self.output(x))
            return x
    
    # Create model
    model = SimpleNet()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Training loop
    for epoch in range(1000):
        # Forward
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """)
    print()
    
    print("🚀 Why PyTorch Makes This Easy:")
    print()
    print("   Our scratch version: ~100 lines of math")
    print("   PyTorch version: ~15 lines")
    print()
    print("   PyTorch handles:")
    print("   • Gradient computation (autograd)")
    print("   • Weight initialization (smart defaults)")
    print("   • Optimization (Adam, SGD, etc.)")
    print("   • GPU acceleration (automatic)")
    print()
    print("   You focus on:")
    print("   • Network architecture (layers, activations)")
    print("   • Training loop logic")
    print("   • Experimentation (try new ideas fast!)")


# ============================================================================
# 4. Activation Functions
# ============================================================================

def demo_activations():
    """
    Activation Functions: Add non-linearity to neural networks
    
    INTUITION - The Decision-Making Process:
    
    Without activation (linear):
    "If score > 0, slightly positive; if score < 0, slightly negative"
    → Boring! Just a straight line, can't solve complex problems
    
    With activation (non-linear):
    "If score > 0, YES! If score < 0, NO!"
    → Exciting! Can make complex decisions, solve hard problems
    
    Think of activations as decision styles:
    
    1. ReLU (Rectified Linear Unit):
       "If negative, ignore it. If positive, keep it."
       Like a bouncer: "Under 21? Out! Over 21? Welcome!"
       Use: Hidden layers (90% of the time)
    
    2. Sigmoid:
       "Squash everything between 0 and 1"
       Like a probability: "How confident are we? 0.8 = 80% sure"
       Use: Binary classification output
    
    3. Tanh:
       "Squash between -1 and +1"
       Like sentiment: "-0.8 = very negative, +0.8 = very positive"
       Use: When you need negative outputs
    
    4. Softmax:
       "Convert scores to probabilities that sum to 1"
       Like voting: "40% cat, 35% dog, 25% bird"
       Use: Multi-class classification
    
    WHY NEEDED?
    
    Without activations:
    Network of 10 layers = Still just a fancy linear model
    Can only draw straight lines to separate data
    
    With activations:
    Network can draw curves, spirals, any shape!
    Can solve complex problems like image recognition
    
    Real Example:
    Linear model: "Is email spam if it has >5 suspicious words?"
    Neural network with activations: "Complex pattern of words,
    sender behavior, timing, links → 95% accurate spam detection"
    """
    print("\n" + "=" * 70)
    print("4. Activation Functions")
    print("=" * 70)
    print()
    print("💭 INTUITION: Different Decision-Making Styles")
    print()
    print("   Without activation (linear):")
    print("   'Score = 2.5 → Output = 2.5'")
    print("   Boring! Just multiplication, can't solve complex problems")
    print()
    print("   With activation (non-linear):")
    print("   'Score = 2.5 → Output = 0.92 (confident YES!)'")
    print("   Exciting! Makes real decisions, solves hard problems")
    print()
    
    print("🎯 Common Activation Functions:")
    print()
    
    # Test values
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    # ReLU
    print("1️⃣  ReLU (Rectified Linear Unit):")
    print("   Formula: f(x) = max(0, x)")
    print("   Intuition: 'Keep positive, zero out negative'")
    print()
    print("   Like a bouncer at a club:")
    print("   'Negative score? Rejected (→ 0)'")
    print("   'Positive score? Approved (→ keep it)'")
    print()
    print("   Values:")
    for x in test_values:
        relu_val = max(0, x)
        print(f"     f({x:5.1f}) = {relu_val:5.1f}")
    print()
    print("   ✅ Use: Hidden layers (fast, works well)")
    print("   ❌ Avoid: Output layer (can't give probabilities)")
    print()
    
    # Sigmoid
    print("2️⃣  Sigmoid:")
    print("   Formula: f(x) = 1 / (1 + e^-x)")
    print("   Intuition: 'Squash to probability (0 to 1)'")
    print()
    print("   Like confidence level:")
    print("   'Very negative → 0 (0% confident)'")
    print("   'Zero → 0.5 (50% confident)'")
    print("   'Very positive → 1 (100% confident)'")
    print()
    print("   Values:")
    for x in test_values:
        sigmoid_val = 1 / (1 + math.exp(-x))
        print(f"     f({x:5.1f}) = {sigmoid_val:5.3f}")
    print()
    print("   ✅ Use: Binary classification output")
    print("   ❌ Avoid: Hidden layers (vanishing gradients)")
    print()
    
    # Tanh
    print("3️⃣  Tanh (Hyperbolic Tangent):")
    print("   Formula: f(x) = (e^x - e^-x) / (e^x + e^-x)")
    print("   Intuition: 'Squash to range (-1 to 1)'")
    print()
    print("   Like sentiment score:")
    print("   'Very negative → -1 (strongly disagree)'")
    print("   'Zero → 0 (neutral)'")
    print("   'Very positive → +1 (strongly agree)'")
    print()
    print("   Values:")
    for x in test_values:
        tanh_val = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        print(f"     f({x:5.1f}) = {tanh_val:6.3f}")
    print()
    print("   ✅ Use: When you need negative outputs")
    print("   ❌ Avoid: Hidden layers (vanishing gradients)")
    print()
    
    # Softmax
    print("4️⃣  Softmax:")
    print("   Formula: f(x_i) = e^x_i / Σ(e^x_j)")
    print("   Intuition: 'Convert scores to probabilities'")
    print()
    print("   Like voting results:")
    print("   Scores: [3.0, 1.0, 0.2] → Probabilities: [0.71, 0.21, 0.08]")
    print("   Sum = 1.0 (100% probability distributed)")
    print()
    scores = [3.0, 1.0, 0.2]
    exp_scores = [math.exp(x) for x in scores]
    sum_exp = sum(exp_scores)
    softmax_probs = [x / sum_exp for x in exp_scores]
    
    print("   Example:")
    print(f"     Input scores: {scores}")
    print(f"     Probabilities: {[f'{p:.3f}' for p in softmax_probs]}")
    print(f"     Sum: {sum(softmax_probs):.3f} ✓")
    print()
    print("   ✅ Use: Multi-class classification output")
    print("   Interpretation: 'Cat: 71%, Dog: 21%, Bird: 8%'")
    print()
    
    print("💡 DECISION GUIDE:")
    print()
    print("   Hidden Layers:")
    print("   → ReLU (default choice, works 90% of time)")
    print("   → Leaky ReLU (if ReLU gives 'dead neurons')")
    print()
    print("   Output Layer:")
    print("   → Sigmoid (binary: yes/no, spam/not spam)")
    print("   → Softmax (multi-class: cat/dog/bird)")
    print("   → Linear (regression: predict house price)")
    print()
    
    print("🧠 WHY ACTIVATIONS ARE CRITICAL:")
    print()
    print("   Without activations:")
    print("   • 10-layer network = Still just linear model")
    print("   • Can only draw straight lines")
    print("   • Can't solve XOR, image recognition, NLP")
    print()
    print("   With activations:")
    print("   • Can approximate any function!")
    print("   • Draw complex decision boundaries")
    print("   • Solve real-world problems")
    print()
    print("   Example:")
    print("   Linear: 67% accuracy on image classification")
    print("   Deep network with ReLU: 95% accuracy!")
    print()
    
    print("⚠️  Common Pitfalls:")
    print()
    print("   1️⃣  Using sigmoid in hidden layers:")
    print("      Problem: Vanishing gradients (training stops)")
    print("      Fix: Use ReLU")
    print()
    print("   2️⃣  Forgetting activation in output:")
    print("      Problem: Raw scores, not probabilities")
    print("      Fix: Add sigmoid (binary) or softmax (multi-class)")
    print()
    print("   3️⃣  Using softmax for binary classification:")
    print("      Problem: Overkill, slower")
    print("      Fix: Use sigmoid (simpler, faster)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🔥 PyTorch Basics for Deep Learning\n")
    print("Learn how PyTorch makes deep learning accessible!")
    print()
    
    demo_tensors()
    demo_autograd()
    demo_neural_network()
    demo_activations()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Tensors: Multi-dimensional arrays (GPU-accelerated)
   - 0D: Scalar (loss value)
   - 1D: Vector (embeddings)
   - 2D: Matrix (grayscale image)
   - 3D: RGB image
   - 4D: Batch of images

2. Autograd: Automatic gradient computation
   - No manual backprop math needed!
   - Call .backward() → All gradients computed
   - Enables rapid experimentation

3. Neural Networks: Stack of transformations
   - Hidden layers: Extract features
   - Activations: Add non-linearity
   - Output layer: Make predictions

4. Activation Functions:
   - ReLU: Hidden layers (default choice)
   - Sigmoid: Binary classification output
   - Softmax: Multi-class output

Real Impact:
- Train models 100x faster (GPU)
- Build complex architectures in minutes (autograd)
- Focus on creativity, not calculus!

Next Steps:
- 02_simple_cnn.py: Build image classifier
- 03_transfer_learning.py: Use pre-trained models
- Practice: Implement different architectures!
""")


if __name__ == "__main__":
    main()
