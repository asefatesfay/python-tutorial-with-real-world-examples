"""
Calculus Essentials for ML/AI Engineering

Build intuition for derivatives and gradient descent - how models learn!
Focus: Understanding optimization, not complex math proofs.

Install: poetry add numpy matplotlib
Run: poetry run python 03-math-for-ml/examples/03_calculus_essentials.py
"""

from typing import Callable, List, Tuple
import random


# ============================================================================
# 1. Derivatives - Rate of Change
# ============================================================================

def demo_derivatives():
    """
    Derivative: How fast something changes.
    
    INTUITION - Your Car's Speedometer:
    
    You're on a road trip:
    - Position: Mile marker 50 (where you are)
    - Speed: 60 mph (how fast position changes) ← THIS IS THE DERIVATIVE!
    - Acceleration: +5 mph/s (speeding up) ← derivative of speed
    
    Derivative = "Rate of change" = "How fast is X changing?"
    
    Speed tells you:
    - Positive speed → Moving forward ↗
    - Negative speed → Moving backward ↙  
    - Zero speed → Standing still —
    - High speed → Changing position quickly
    
    In ML - Same Concept:
    - Position → Your model's loss (error)
    - Speed → Derivative of loss (how loss changes with weights)
    - We want: "Which way should I adjust weights to reduce loss?"
    
    Derivative tells us:
    - Positive → Loss increasing (bad direction!) ↗
    - Negative → Loss decreasing (good direction!) ↙
    - Zero → At minimum or maximum (flat spot) —
    
    ML Use:
    - How does loss change when we adjust weights?
    - Which direction improves the model?
    """
    print("=" * 70)
    print("1. Derivatives - Rate of Change")
    print("=" * 70)
    print()
    print("💭 INTUITION: Your Car's Speedometer")
    print()
    print("   Position: Mile 50 (where you are)")
    print("   Speed: 60 mph (HOW FAST position changes) ← Derivative!")
    print("   Acceleration: +5 mph/s (how fast speed changes)")
    print()
    print("   In ML:")
    print("   Loss: 0.42 (model error)")
    print("   Gradient: -0.05 (how fast loss changes) ← Derivative!")
    print("   → Negative gradient = loss decreasing = good!")
    print()
    print("   Speed tells you direction on road")
    print("   Gradient tells you direction to improve model")
    print()
    
    # Simple function: f(x) = x²
    def f(x: float) -> float:
        """Simple quadratic function."""
        return x ** 2
    
    # Derivative: f'(x) = 2x (how fast f(x) changes)
    def f_derivative(x: float) -> float:
        """Derivative of f(x) = x²."""
        return 2 * x
    
    print("Function: f(x) = x²")
    print("Derivative: f'(x) = 2x")
    print()
    
    test_points = [-2, -1, 0, 1, 2]
    
    print("x  | f(x) | f'(x) | Interpretation")
    print("---|------|-------|----------------")
    for x in test_points:
        fx = f(x)
        fpx = f_derivative(x)
        
        if fpx > 0:
            direction = "going up ↗"
        elif fpx < 0:
            direction = "going down ↘"
        else:
            direction = "flat —"
        
        print(f"{x:2d} | {fx:4d} | {fpx:5d} | {direction}")
    
    print()
    print("💡 Derivative tells us:")
    print("   - f'(x) > 0: Function increasing")
    print("   - f'(x) < 0: Function decreasing")
    print("   - f'(x) = 0: Flat (might be min/max!)")
    print()
    
    # Numerical derivative (approximation)
    def numerical_derivative(func: Callable, x: float, h: float = 0.0001) -> float:
        """
        Approximate derivative using small step.
        
        Formula: f'(x) ≈ (f(x+h) - f(x)) / h
        
        This is how computers calculate derivatives!
        """
        return (func(x + h) - func(x)) / h
    
    print("Numerical Approximation:")
    x = 3
    exact = f_derivative(x)
    approx = numerical_derivative(f, x)
    print(f"  At x = {x}:")
    print(f"  Exact: f'({x}) = {exact}")
    print(f"  Approx: f'({x}) ≈ {approx:.4f}")
    print(f"  Error: {abs(exact - approx):.6f}")
    print()
    
    print("💡 In ML:")
    print("   Libraries (PyTorch, TensorFlow) compute derivatives automatically!")
    print("   This is called 'automatic differentiation' or 'autograd'")


# ============================================================================
# 2. Gradient Descent - How Models Learn
# ============================================================================

def demo_gradient_descent():
    """
    Gradient Descent: Algorithm to find minimum of a function.
    
    INTUITION - Lost Hiker in Foggy Mountains:
    
    You're hiking in thick fog, want to reach the valley (lowest point).
    Can't see anything! But you can:
    1. Feel ground with your feet → Which way slopes down?
    2. Take a step downhill
    3. Repeat until ground is flat (you're at the bottom!)
    
    This is EXACTLY how ML models learn:
    - Mountain height = Loss (model error)
    - Your position = Model weights
    - Valley (lowest point) = Best weights (minimum loss)
    - Feeling slope = Computing gradient (derivative)
    - Taking step downhill = Updating weights
    
    Process:
    1. Start somewhere random (initialize weights)
    2. Feel slope (compute gradient)
    3. Step downhill (weights = weights - learning_rate × gradient)
    4. Repeat 10,000 times → Reach bottom (trained model!)
    
    WHY "GRADIENT DESCENT"?
    - Gradient = Slope direction (which way is uphill)
    - Descent = We go opposite direction (downhill)
    - Result = We find the minimum (best model)
    
    Every ML training uses this! GPT, image recognition, everything.
    
    ML Use:
    - Find weights that minimize loss
    - Training neural networks
    - Optimizing any ML model
    """
    print("\n" + "=" * 70)
    print("2. Gradient Descent - How Models Learn")
    print("=" * 70)
    print()
    print("💭 INTUITION: Lost Hiker in Foggy Mountains")
    print()
    print("   You're in thick fog, want to reach valley (lowest point).")
    print("   Can't see, but can feel which way slopes down!")
    print()
    print("   1. Feel ground → Slopes down to the left")
    print("   2. Step left (downhill)")
    print("   3. Feel again → Still slopes left")
    print("   4. Keep going → Ground becomes flat → You're at bottom! ✓")
    print()
    print("   ML Training = Same Process:")
    print("   - Mountain = Loss (error)")
    print("   - Valley = Best weights (minimum error)")
    print("   - Walking downhill = Adjusting weights")
    print("   - Reaching bottom = Trained model!")
    print()
    print("   This is how ChatGPT, image recognition, ALL models learn!")
    print()
    
    # Function to minimize: f(x) = (x - 3)² + 5
    # Minimum at x = 3, f(3) = 5
    def loss_function(x: float) -> float:
        """Loss function (we want to minimize this)."""
        return (x - 3) ** 2 + 5
    
    def loss_derivative(x: float) -> float:
        """Derivative of loss function."""
        return 2 * (x - 3)
    
    print("Loss function: L(x) = (x - 3)² + 5")
    print("Minimum at: x = 3, L(3) = 5")
    print()
    
    # Gradient descent
    x = 0.0  # Start position (random guess)
    learning_rate = 0.1  # Step size
    iterations = 20
    
    print("Gradient Descent:")
    print()
    print("Step | x      | L(x)   | L'(x)  | Update")
    print("-----|--------|--------|--------|----------------")
    
    for step in range(iterations):
        loss = loss_function(x)
        gradient = loss_derivative(x)
        
        # Gradient descent update: x_new = x_old - learning_rate * gradient
        x_new = x - learning_rate * gradient
        
        if step < 10 or step >= iterations - 2:  # Show first 10 and last 2
            print(f"{step:4d} | {x:6.3f} | {loss:6.3f} | {gradient:6.3f} | "
                  f"x = {x:.3f} - {learning_rate}*{gradient:.3f} = {x_new:.3f}")
        elif step == 10:
            print("  ... (iterations 10-17)")
        
        x = x_new
        
        # Stop if gradient is very small (converged)
        if abs(gradient) < 0.001:
            print(f"\n✅ Converged at step {step}!")
            break
    
    print()
    print(f"Final result: x = {x:.3f}, L(x) = {loss_function(x):.3f}")
    print(f"True minimum: x = 3.000, L(x) = 5.000")
    print()
    
    print("💡 In ML:")
    print("   - x: Model weights")
    print("   - L(x): Loss (error)")
    print("   - L'(x): Gradient (how to improve)")
    print("   - Update: weights = weights - learning_rate * gradient")


# ============================================================================
# 3. Learning Rate - Critical Hyperparameter
# ============================================================================

def demo_learning_rate():
    """
    Learning Rate: How big each step is in gradient descent.
    
    INTUITION - Walking Down Stairs in the Dark:
    
    You're walking down stairs but can't see. How big should your steps be?
    
    TOO SMALL (lr = 0.001):
    - Baby steps: 1 inch at a time
    - Super safe, but takes FOREVER to get down
    - Problem: Might give up before reaching bottom
    
    JUST RIGHT (lr = 0.1):
    - Normal steps: One stair at a time
    - Safe and efficient
    - Reaches bottom in reasonable time ✓
    
    TOO LARGE (lr = 1.0):
    - Giant leaps: 5 stairs at once
    - Miss steps, bounce back and forth
    - Might fall! (training diverges)
    - Or jump over the bottom entirely!
    
    In ML Training:
    - Too small: Training takes days/weeks, costs $$$$
    - Just right: Model converges in hours ✓
    - Too large: Loss explodes to infinity, model broken ✗
    
    REAL IMPACT:
    - Wrong learning rate = Main reason training fails
    - Common values: 0.001, 0.0001 (try different ones!)
    - Advanced: Start large, decrease over time ("learning rate schedule")
    
    ML Use:
    - Most important hyperparameter
    - Affects training speed and success
    """
    print("\n" + "=" * 70)
    print("3. Learning Rate - Critical Hyperparameter")
    print("=" * 70)
    print()
    print("💭 INTUITION: Walking Down Stairs in the Dark")
    print()
    print("   How big should each step be?")
    print()
    print("   TOO SMALL (lr=0.001): Baby steps")
    print("   → Takes forever! Might never reach bottom.")
    print()
    print("   JUST RIGHT (lr=0.1): Normal steps")
    print("   → One stair at a time. Safe & efficient! ✓")
    print()
    print("   TOO LARGE (lr=1.0): Giant leaps")
    print("   → Skip stairs, bounce around, might fall! ✗")
    print()
    print("   In ML: Wrong learning rate = #1 reason training fails!")
    print("   Start with 0.001 or 0.0001, adjust if needed.")
    print()
    
    def loss_function(x: float) -> float:
        return (x - 3) ** 2 + 5
    
    def loss_derivative(x: float) -> float:
        return 2 * (x - 3)
    
    # Test different learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.1]
    
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        print("-" * 50)
        
        x = 0.0
        max_iters = 50
        
        for step in range(max_iters):
            gradient = loss_derivative(x)
            x_new = x - lr * gradient
            
            # Show first few steps
            if step < 5:
                print(f"  Step {step}: x = {x:.3f} → {x_new:.3f}")
            
            # Check for divergence
            if abs(x_new) > 100:
                print(f"  ❌ Diverged at step {step}! x = {x_new:.3f}")
                break
            
            # Check for convergence
            if abs(gradient) < 0.001:
                print(f"  ✅ Converged at step {step}! x = {x:.3f}")
                break
            
            x = x_new
        else:
            print(f"  ⚠️  Did not converge in {max_iters} steps. x = {x:.3f}")
    
    print("\n💡 Key insights:")
    print("   lr = 0.01: Too small, slow convergence")
    print("   lr = 0.1: Good, converges quickly")
    print("   lr = 0.5: Still okay, slightly oscillating")
    print("   lr = 1.1: Too large, diverges!")
    print()
    print("In practice:")
    print("   - Start with lr = 0.001 or 0.01")
    print("   - Use learning rate schedules (decrease over time)")
    print("   - Try different values (hyperparameter tuning)")


# ============================================================================
# 4. Partial Derivatives - Multiple Variables
# ============================================================================

def demo_partial_derivatives():
    """
    Partial Derivative: Derivative with respect to one variable.
    
    INTUITION - Room Temperature Control:
    
    Your room comfort depends on TWO things:
    - Heater setting (x): 0-100%
    - AC setting (y): 0-100%
    - Comfort: f(x, y) = some function of both
    
    Partial derivatives answer:
    - ∂f/∂x: "If I turn UP heater (keep AC fixed), does comfort improve?"
    - ∂f/∂y: "If I turn UP AC (keep heater fixed), does comfort improve?"
    
    Example: You're cold
    - ∂comfort/∂heater = +5 (turning up heater helps a lot!)
    - ∂comfort/∂AC = -3 (turning up AC makes it worse)
    - Action: Increase heater, don't touch AC
    
    In ML with millions of weights:
    - Weight 1: ∂loss/∂w1 = -0.05 (decrease w1 to reduce loss)
    - Weight 2: ∂loss/∂w2 = +0.03 (increase w2 to reduce loss)
    - Weight 3: ∂loss/∂w3 = 0.00 (this weight doesn't matter right now)
    - ... (millions more)
    
    Gradient = Vector of ALL partial derivatives
    - Tells you which "direction" to adjust all weights simultaneously
    - This is what backpropagation computes!
    
    THE MAGIC:
    One forward pass + one backward pass = gradients for ALL weights
    Even if you have 175 billion parameters (like GPT-3)!
    
    Function: f(x, y) = x² + 2xy + y²
    ∂f/∂x: How f changes when x changes (y fixed)
    ∂f/∂y: How f changes when y changes (x fixed)
    
    ML Use:
    - Neural networks have millions of weights
    - Need derivative for each weight
    - Gradient is vector of all partial derivatives
    """
    print("\n" + "=" * 70)
    print("4. Partial Derivatives - Multiple Variables")
    print("=" * 70)
    print()
    print("💭 INTUITION: Room Temperature Control")
    print()
    print("   Comfort depends on TWO controls:")
    print("   - Heater: 0-100%")
    print("   - AC: 0-100%")
    print()
    print("   You're cold. What should you adjust?")
    print()
    print("   ∂comfort/∂heater = +5 (heater helps!)")
    print("   ∂comfort/∂AC = -3 (AC makes it worse)")
    print()
    print("   Action: Turn up heater, leave AC alone.")
    print()
    print("   In ML:")
    print("   ∂loss/∂weight1 = -0.05 (adjust this weight!)")
    print("   ∂loss/∂weight2 = +0.03 (adjust this too!)")
    print("   ... for MILLIONS of weights")
    print()
    print("   Gradient = All partial derivatives together")
    print("   Backpropagation = Computing all of them efficiently!")
    print()
    
    # Function: f(x, y) = x² + y²
    # This is a bowl shape, minimum at (0, 0)
    def f(x: float, y: float) -> float:
        """Bowl-shaped function."""
        return x**2 + y**2
    
    # Partial derivatives
    def df_dx(x: float, y: float) -> float:
        """Partial derivative with respect to x."""
        return 2 * x
    
    def df_dy(x: float, y: float) -> float:
        """Partial derivative with respect to y."""
        return 2 * y
    
    print("Function: f(x, y) = x² + y²")
    print("Minimum at: (0, 0), f(0,0) = 0")
    print()
    print("Partial derivatives:")
    print("  ∂f/∂x = 2x")
    print("  ∂f/∂y = 2y")
    print()
    
    # Gradient descent with multiple variables
    x, y = 5.0, 3.0  # Starting point
    lr = 0.1
    
    print("Gradient Descent (2D):")
    print()
    print("Step | x      | y      | f(x,y) | Gradient")
    print("-----|--------|--------|--------|------------------")
    
    for step in range(15):
        loss = f(x, y)
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        
        if step < 8 or step >= 13:
            print(f"{step:4d} | {x:6.3f} | {y:6.3f} | {loss:6.3f} | "
                  f"({grad_x:6.3f}, {grad_y:6.3f})")
        elif step == 8:
            print("  ...")
        
        # Update both variables
        x = x - lr * grad_x
        y = y - lr * grad_y
        
        # Check convergence
        if abs(grad_x) < 0.01 and abs(grad_y) < 0.01:
            print(f"\n✅ Converged!")
            break
    
    print()
    print(f"Final: ({x:.3f}, {y:.3f}), f = {f(x, y):.3f}")
    print(f"True minimum: (0.000, 0.000), f = 0.000")
    print()
    
    print("💡 In neural networks:")
    print("   - Millions of weights (w₁, w₂, ..., wₙ)")
    print("   - Compute ∂Loss/∂w₁, ∂Loss/∂w₂, ..., ∂Loss/∂wₙ")
    print("   - Update all weights simultaneously")
    print("   - This is done automatically by PyTorch/TensorFlow!")


# ============================================================================
# 5. Chain Rule - Backpropagation
# ============================================================================

def demo_chain_rule():
    """
    Chain Rule: Compute derivative of composed functions.
    
    Formula: If y = f(g(x)), then dy/dx = df/dg * dg/dx
    
    ML Use:
    - Neural networks are composed functions (layers)
    - Backpropagation uses chain rule
    - Compute gradient layer by layer backwards
    """
    print("\n" + "=" * 70)
    print("5. Chain Rule - Backpropagation")
    print("=" * 70)
    
    print("🎯 Simple neural network (1 layer):")
    print()
    print("   Input (x) → Linear (z = wx + b) → Activation (a = σ(z)) → Loss (L)")
    print()
    print("   Want: ∂L/∂w (how loss changes with weight)")
    print()
    print("   Chain rule: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w")
    print()
    
    # Example: Single neuron
    # Input
    x = 2.0
    
    # Weight and bias
    w = 0.5
    b = 0.1
    
    # True target
    y_true = 3.0
    
    print("Forward Pass:")
    print(f"  Input: x = {x}")
    print(f"  Weight: w = {w}")
    print(f"  Bias: b = {b}")
    print()
    
    # Linear layer
    z = w * x + b
    print(f"  Linear: z = w*x + b = {w}*{x} + {b} = {z}")
    
    # Activation (ReLU: max(0, z))
    a = max(0, z)
    print(f"  ReLU: a = max(0, z) = max(0, {z}) = {a}")
    
    # Loss (MSE: (y_true - y_pred)²)
    loss = (y_true - a) ** 2
    print(f"  Loss: L = (y_true - a)² = ({y_true} - {a})² = {loss:.3f}")
    print()
    
    print("Backward Pass (Chain Rule):")
    print()
    
    # ∂L/∂a
    dL_da = -2 * (y_true - a)
    print(f"  ∂L/∂a = -2(y_true - a) = -2({y_true} - {a}) = {dL_da:.3f}")
    
    # ∂a/∂z (ReLU derivative: 1 if z > 0, else 0)
    da_dz = 1.0 if z > 0 else 0.0
    print(f"  ∂a/∂z = 1 (since z > 0)")
    
    # ∂z/∂w
    dz_dw = x
    print(f"  ∂z/∂w = x = {x}")
    print()
    
    # Chain rule: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w
    dL_dw = dL_da * da_dz * dz_dw
    print(f"  ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w")
    print(f"        = {dL_da:.3f} * {da_dz:.3f} * {dz_dw:.3f}")
    print(f"        = {dL_dw:.3f}")
    print()
    
    # Update weight
    lr = 0.1
    w_new = w - lr * dL_dw
    print(f"  Update: w_new = w - lr * ∂L/∂w")
    print(f"                = {w} - {lr} * {dL_dw:.3f}")
    print(f"                = {w_new:.3f}")
    print()
    
    # Verify improvement
    z_new = w_new * x + b
    a_new = max(0, z_new)
    loss_new = (y_true - a_new) ** 2
    
    print(f"  Old loss: {loss:.3f}")
    print(f"  New loss: {loss_new:.3f}")
    print(f"  Improvement: {loss - loss_new:.3f} ✅")
    print()
    
    print("💡 In deep networks:")
    print("   - Apply chain rule through all layers")
    print("   - Compute gradients backwards (backpropagation)")
    print("   - PyTorch/TensorFlow do this automatically!")


# ============================================================================
# 6. Real-World: Training a Simple Model
# ============================================================================

def demo_training():
    """
    Real-world ML: Train a model to fit data.
    
    Model: y = w * x + b (linear regression)
    Goal: Find w and b that best fit the data
    Method: Gradient descent on MSE loss
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Training a Linear Model")
    print("=" * 70)
    
    print("📊 Problem: Predict house price from size")
    print()
    
    # Training data (size in 1000 sqft, price in $100k)
    data = [
        (1.0, 2.0),  # 1000 sqft → $200k
        (2.0, 4.0),  # 2000 sqft → $400k
        (3.0, 6.0),  # 3000 sqft → $600k
        (4.0, 8.0),  # 4000 sqft → $800k
    ]
    
    print("Training data:")
    for size, price in data:
        print(f"  {size*1000:.0f} sqft → ${price*100:.0f}k")
    print()
    
    # True relationship: price = 2 * size (we'll try to learn this)
    # Model: y = w * x + b
    
    # Initialize weights randomly
    w = random.uniform(0, 1)
    b = random.uniform(0, 1)
    
    lr = 0.01
    epochs = 100
    
    print(f"Initial: w = {w:.3f}, b = {b:.3f}")
    print()
    print("Training...")
    print()
    
    for epoch in range(epochs):
        total_loss = 0
        dw = 0  # Gradient for w
        db = 0  # Gradient for b
        
        # Compute loss and gradients over all data
        for x, y_true in data:
            # Forward pass
            y_pred = w * x + b
            
            # Loss (MSE)
            loss = (y_true - y_pred) ** 2
            total_loss += loss
            
            # Gradients (from calculus)
            # ∂L/∂w = -2(y_true - y_pred) * x
            # ∂L/∂b = -2(y_true - y_pred)
            dw += -2 * (y_true - y_pred) * x
            db += -2 * (y_true - y_pred)
        
        # Average gradients
        dw /= len(data)
        db /= len(data)
        avg_loss = total_loss / len(data)
        
        # Update weights
        w = w - lr * dw
        b = b - lr * db
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, w = {w:.3f}, b = {b:.3f}")
    
    print()
    print(f"Final model: y = {w:.3f} * x + {b:.3f}")
    print(f"True model:  y = 2.000 * x + 0.000")
    print()
    
    # Test predictions
    print("Predictions:")
    test_sizes = [1.5, 2.5, 3.5]
    for size in test_sizes:
        pred = w * size + b
        true_val = 2.0 * size  # True relationship
        print(f"  {size*1000:.0f} sqft → ${pred*100:.0f}k (true: ${true_val*100:.0f}k)")
    
    print()
    print("💡 This is how all ML models learn:")
    print("   1. Make predictions (forward pass)")
    print("   2. Compute loss (error)")
    print("   3. Compute gradients (calculus!)")
    print("   4. Update weights (gradient descent)")
    print("   5. Repeat until converged")


# ============================================================================
# 7. Common Activation Functions and Derivatives
# ============================================================================

def demo_activation_functions():
    """
    Activation functions: Add non-linearity to neural networks.
    
    Without activation: Neural network = glorified linear regression
    With activation: Can learn complex patterns
    
    Common activations:
    - ReLU: max(0, x)
    - Sigmoid: 1 / (1 + e^-x)
    - Tanh: (e^x - e^-x) / (e^x + e^-x)
    """
    print("\n" + "=" * 70)
    print("7. Activation Functions and Their Derivatives")
    print("=" * 70)
    
    import math
    
    # ReLU
    def relu(x: float) -> float:
        return max(0, x)
    
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    # Sigmoid
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(x: float) -> float:
        s = sigmoid(x)
        return s * (1 - s)
    
    # Tanh
    def tanh(x: float) -> float:
        return math.tanh(x)
    
    def tanh_derivative(x: float) -> float:
        t = tanh(x)
        return 1 - t**2
    
    print("ReLU: f(x) = max(0, x)")
    print("  Used in: Most modern neural networks")
    print("  Why: Fast, simple, works well")
    print()
    
    test_values = [-2, -1, 0, 1, 2]
    print("x  | ReLU | ReLU'")
    print("---|------|-------")
    for x in test_values:
        print(f"{x:2d} | {relu(x):4.1f} | {relu_derivative(x):5.1f}")
    
    print()
    print("Sigmoid: f(x) = 1/(1 + e^-x)")
    print("  Used in: Binary classification, gates (LSTM)")
    print("  Output: (0, 1) - probability-like")
    print()
    
    print("x  | Sigmoid | Sigmoid'")
    print("---|---------|----------")
    for x in test_values:
        print(f"{x:2d} | {sigmoid(x):7.3f} | {sigmoid_derivative(x):8.3f}")
    
    print()
    print("Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x)")
    print("  Used in: RNNs, sometimes instead of sigmoid")
    print("  Output: (-1, 1) - zero-centered")
    print()
    
    print("x  | Tanh   | Tanh'")
    print("---|--------|-------")
    for x in test_values:
        print(f"{x:2d} | {tanh(x):6.3f} | {tanh_derivative(x):6.3f}")
    
    print()
    print("💡 Key insight:")
    print("   - Derivative needed for backpropagation")
    print("   - ReLU: Simple derivative (0 or 1)")
    print("   - Sigmoid/Tanh: More complex, can vanish for large |x|")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n📐 Calculus Essentials for ML/AI Engineering\n")
    print("Focus: Understanding optimization and how models learn!")
    print()
    
    demo_derivatives()
    demo_gradient_descent()
    demo_learning_rate()
    demo_partial_derivatives()
    demo_chain_rule()
    demo_training()
    demo_activation_functions()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Derivative: Rate of change
   - f'(x) > 0: Increasing
   - f'(x) < 0: Decreasing
   - f'(x) = 0: Flat (min/max)

2. Gradient Descent: Optimization algorithm
   - Start at random point
   - Move opposite to gradient
   - Repeat until minimum

3. Learning Rate: Step size
   - Too small: Slow training
   - Too large: Diverge
   - Just right: Fast convergence

4. Partial Derivatives: Multiple variables
   - Gradient = vector of partial derivatives
   - Update all weights together

5. Chain Rule: Backpropagation
   - Compute gradients layer by layer
   - Go backwards through network
   - Automatic in PyTorch/TensorFlow

6. Training Loop:
   - Forward pass: Predictions
   - Compute loss: Error
   - Backward pass: Gradients
   - Update weights: Gradient descent

ML Applications:
- All model training uses gradient descent
- Backpropagation uses chain rule
- Learning rate most important hyperparameter
- Activation functions need derivatives

Next: 04_information_theory.py - Entropy, cross-entropy, loss functions!
""")


if __name__ == "__main__":
    main()
