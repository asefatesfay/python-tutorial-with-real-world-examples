"""
Linear Regression From Scratch

Build linear regression from the ground up to understand how ML really works!
Focus: Intuition first, then implementation, then optimization.

No libraries needed for core implementation (we'll compare to sklearn at the end)
Run: poetry run python 06-machine-learning-fundamentals/from_scratch/01_linear_regression.py
"""

import random
from typing import List, Tuple


# ============================================================================
# 1. What is Linear Regression? The Intuition
# ============================================================================

def demo_linear_regression_intuition():
    """
    Linear Regression: Find the best straight line through data points.
    
    INTUITION - The House Price Example:
    
    You're a real estate agent trying to price houses.
    You notice: Bigger houses cost more (duh!)
    
    Data points:
    - 1000 sqft → $200k
    - 1500 sqft → $300k
    - 2000 sqft → $400k
    
    Question: What should a 1750 sqft house cost?
    
    Linear Regression finds: Price = a * Size + b
    - a (slope): How much price increases per sqft
    - b (intercept): Base price (when size = 0)
    
    THE MAGIC:
    Instead of guessing, we use math to find the BEST line!
    - Try line 1: Some houses overpriced, some underpriced
    - Try line 2: Even worse fit!
    - Try line 3: Perfect! Minimal errors across all houses
    
    HOW WE FIND IT:
    1. Start with random line (bad predictions)
    2. Calculate how wrong we are (error/loss)
    3. Adjust line to reduce error
    4. Repeat until error stops decreasing
    
    This is called "Gradient Descent" - we'll see it in action!
    
    REAL-WORLD USE CASES:
    - Predict salary from years of experience
    - Predict sales from advertising budget
    - Predict temperature from time of day
    - Any problem where output increases/decreases linearly with input
    
    KEY ASSUMPTIONS:
    ✅ Relationship is roughly linear (straight line makes sense)
    ✅ One feature affects the outcome
    ⚠️  Real world is often messier (we'll handle that later!)
    """
    print("=" * 70)
    print("1. Linear Regression Intuition")
    print("=" * 70)
    print()
    print("💭 INTUITION: The House Pricing Problem")
    print()
    print("   You're pricing houses based on size:")
    print("   • 1000 sqft → $200,000")
    print("   • 1500 sqft → $300,000")
    print("   • 2000 sqft → $400,000")
    print()
    print("   ❓ What should 1750 sqft house cost?")
    print()
    print("   Linear Regression finds the best line:")
    print("   Price = (slope × Size) + intercept")
    print("   Price = (100 × Size) + 100,000")
    print()
    print("   For 1750 sqft:")
    print("   Price = (100 × 1750) + 100,000 = $275,000 ✓")
    print()
    print("🎯 THE GOAL:")
    print("   Find slope & intercept that minimize prediction errors")
    print("   across ALL data points!")
    print()
    print("📊 Real-World Examples:")
    print("   • Salary = slope × YearsExperience + base_salary")
    print("   • Sales = slope × AdBudget + baseline_sales")
    print("   • Test_Score = slope × StudyHours + natural_ability")
    print()


# ============================================================================
# 2. The Math: Cost Function (Mean Squared Error)
# ============================================================================

def demo_cost_function():
    """
    Cost Function: Measures how bad our predictions are.
    
    INTUITION - The Dartboard Analogy:
    
    Imagine throwing darts at a target:
    - Bullseye = perfect prediction
    - Off by 1 inch = small error
    - Off by 10 inches = big error
    
    We want to measure total "wrongness" across all throws.
    
    MEAN SQUARED ERROR (MSE):
    For each prediction:
    1. Calculate error = actual - predicted
    2. Square it (makes all errors positive, penalizes big errors more)
    3. Average all squared errors
    
    WHY SQUARE?
    - error = +5 or -5 are both equally bad
    - Squaring: Both become 25 (positive)
    - Big errors hurt more: 10² = 100 vs 2² = 4
    
    EXAMPLE:
    True prices: [$200k, $300k, $400k]
    Predicted:   [$210k, $290k, $420k]
    Errors:      [+10k,  -10k,  +20k]
    Squared:     [100M,  100M,  400M]
    MSE:         (100M + 100M + 400M) / 3 = 200M
    
    GOAL OF LINEAR REGRESSION:
    Find slope & intercept that minimize MSE!
    """
    print("\n" + "=" * 70)
    print("2. Cost Function (Mean Squared Error)")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Dartboard Analogy")
    print()
    print("   Throwing darts at target:")
    print("   • Bullseye = perfect prediction (error = 0)")
    print("   • 1 inch off = small error")
    print("   • 10 inches off = BIG error")
    print()
    print("   We square errors because:")
    print("   • Makes all errors positive (+5 and -5 both bad)")
    print("   • Penalizes big errors more (10² = 100 vs 2² = 4)")
    print()
    
    # Example calculation
    actual = [200, 300, 400]
    predicted = [210, 290, 420]
    
    print("📊 Example: House Price Predictions")
    print()
    print("   House 1: Actual=$200k, Predicted=$210k, Error=+$10k")
    print("   House 2: Actual=$300k, Predicted=$290k, Error=-$10k")
    print("   House 3: Actual=$400k, Predicted=$420k, Error=+$20k")
    print()
    
    # Calculate MSE
    errors = [actual[i] - predicted[i] for i in range(len(actual))]
    squared_errors = [e ** 2 for e in errors]
    mse = sum(squared_errors) / len(squared_errors)
    
    print(f"   Errors: {errors}")
    print(f"   Squared: {squared_errors}")
    print(f"   MSE: {mse:.0f}")
    print()
    print("🎯 GOAL: Find line parameters that minimize MSE!")
    print()


# ============================================================================
# 3. Implementation: Linear Regression Class
# ============================================================================

class LinearRegression:
    """
    Linear Regression from scratch using gradient descent.
    
    Model: y = slope * x + intercept
    
    Gradient Descent:
    1. Start with random slope & intercept
    2. Calculate predictions
    3. Calculate error (MSE)
    4. Update parameters to reduce error
    5. Repeat until convergence
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Initialize linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent (how big each update is)
            iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.slope = 0.0
        self.intercept = 0.0
        self.history = []  # Track loss over time
    
    def predict(self, x: List[float]) -> List[float]:
        """
        Make predictions using current line parameters.
        
        Formula: y = slope * x + intercept
        """
        return [self.slope * xi + self.intercept for xi in x]
    
    def _calculate_loss(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Mean Squared Error.
        
        MSE = average((actual - predicted)²)
        """
        predictions = self.predict(x)
        errors = [(y[i] - predictions[i]) ** 2 for i in range(len(y))]
        return sum(errors) / len(errors)
    
    def _calculate_gradients(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Calculate gradients (how to adjust parameters).
        
        Math:
        - d(slope) = -2 * average(x * error)
        - d(intercept) = -2 * average(error)
        
        Where error = actual - predicted
        """
        n = len(x)
        predictions = self.predict(x)
        errors = [y[i] - predictions[i] for i in range(n)]
        
        # Gradient for slope
        slope_gradient = -2 * sum(x[i] * errors[i] for i in range(n)) / n
        
        # Gradient for intercept
        intercept_gradient = -2 * sum(errors) / n
        
        return slope_gradient, intercept_gradient
    
    def fit(self, x: List[float], y: List[float], verbose: bool = False) -> None:
        """
        Train the model using gradient descent.
        
        Process:
        1. Calculate current loss
        2. Calculate gradients (direction to improve)
        3. Update parameters
        4. Repeat
        """
        # Initialize with random small values
        self.slope = random.uniform(-1, 1)
        self.intercept = random.uniform(-1, 1)
        
        for iteration in range(self.iterations):
            # Calculate current loss
            loss = self._calculate_loss(x, y)
            self.history.append(loss)
            
            # Calculate gradients
            slope_grad, intercept_grad = self._calculate_gradients(x, y)
            
            # Update parameters (move in direction that reduces loss)
            self.slope -= self.learning_rate * slope_grad
            self.intercept -= self.learning_rate * intercept_grad
            
            # Print progress
            if verbose and (iteration % 100 == 0 or iteration == self.iterations - 1):
                print(f"   Iteration {iteration:4d}: Loss={loss:,.0f}, "
                      f"Slope={self.slope:.2f}, Intercept={self.intercept:.0f}")
    
    def score(self, x: List[float], y: List[float]) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_residual / SS_total)
        
        R² = 1.0: Perfect predictions
        R² = 0.0: As good as predicting the mean
        R² < 0.0: Worse than predicting the mean
        """
        predictions = self.predict(x)
        
        # Mean of actual values
        y_mean = sum(y) / len(y)
        
        # Sum of squared residuals (our errors)
        ss_residual = sum((y[i] - predictions[i]) ** 2 for i in range(len(y)))
        
        # Total sum of squares (variance in data)
        ss_total = sum((yi - y_mean) ** 2 for yi in y)
        
        # R² score
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        
        return r2


# ============================================================================
# 4. Demo: Training Linear Regression
# ============================================================================

def demo_training():
    """
    Train linear regression on house price data.
    
    Watch how gradient descent finds the best line!
    """
    print("\n" + "=" * 70)
    print("3. Training Linear Regression")
    print("=" * 70)
    print()
    print("📊 Dataset: House Prices")
    print()
    
    # Simple dataset: house size (sqft) -> price ($1000s)
    sizes = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
    prices = [200, 230, 270, 300, 340, 380, 410, 450]  # In $1000s
    
    print("   Training Data:")
    for size, price in zip(sizes, prices):
        print(f"   {size:4d} sqft → ${price}k")
    print()
    
    # Create and train model
    print("🚀 Training Model (Gradient Descent)...")
    print()
    
    model = LinearRegression(learning_rate=0.0001, iterations=1000)
    model.fit(sizes, prices, verbose=True)
    
    print()
    print("✅ Training Complete!")
    print()
    print(f"   Final Equation: Price = {model.slope:.4f} × Size + {model.intercept:.2f}")
    print()
    
    # Make predictions
    print("🔮 Making Predictions:")
    print()
    
    test_sizes = [1300, 1750, 2100]
    predictions = model.predict(test_sizes)
    
    for size, pred in zip(test_sizes, predictions):
        print(f"   {size:4d} sqft → ${pred:.0f}k predicted")
    print()
    
    # Calculate accuracy
    r2 = model.score(sizes, prices)
    print(f"📈 Model Performance:")
    print(f"   R² Score: {r2:.4f} (1.0 = perfect fit)")
    print()
    
    # Show convergence
    print("📉 Loss Over Time:")
    milestones = [0, 100, 200, 500, 999]
    for i in milestones:
        if i < len(model.history):
            print(f"   Iteration {i:4d}: Loss = {model.history[i]:,.0f}")
    print()
    print("   → Loss decreased! Gradient descent worked! 🎉")
    print()


# ============================================================================
# 5. Understanding Gradient Descent Visually
# ============================================================================

def demo_gradient_descent_intuition():
    """
    Visual understanding of gradient descent.
    
    INTUITION - The Mountain Hiking Analogy:
    
    You're blindfolded on a mountain, trying to reach the valley (lowest point).
    
    Strategy:
    1. Feel the ground slope under your feet
    2. Take a step downhill
    3. Repeat until you can't go lower
    
    In Linear Regression:
    - Mountain = Loss function (higher = worse predictions)
    - Valley = Best parameters (lowest loss)
    - Your position = Current slope & intercept
    - Step size = Learning rate
    
    LEARNING RATE MATTERS:
    - Too small: Takes forever (tiny steps)
    - Too large: Jump over valley, never converge (huge steps)
    - Just right: Reach bottom efficiently
    
    This is the secret behind ALL machine learning!
    """
    print("\n" + "=" * 70)
    print("4. Understanding Gradient Descent")
    print("=" * 70)
    print()
    print("💭 INTUITION: Hiking Down a Mountain (Blindfolded!)")
    print()
    print("   Goal: Reach the valley (lowest point)")
    print()
    print("   Strategy:")
    print("   1. Feel which way is downhill")
    print("   2. Take a step in that direction")
    print("   3. Repeat until you reach the bottom")
    print()
    print("   In Machine Learning:")
    print("   • Mountain height = Loss (prediction error)")
    print("   • Valley = Best parameters")
    print("   • Step size = Learning rate")
    print()
    
    # Demonstrate different learning rates
    print("🎯 Learning Rate Impact:")
    print()
    
    # Same data
    sizes = [1000, 1500, 2000, 2500]
    prices = [200, 300, 400, 500]
    
    learning_rates = [
        (0.00001, "Too Small (Slow)"),
        (0.0001, "Just Right"),
        (0.001, "Too Large (Unstable)")
    ]
    
    for lr, description in learning_rates:
        model = LinearRegression(learning_rate=lr, iterations=500)
        model.fit(sizes, prices, verbose=False)
        
        final_loss = model.history[-1] if model.history else float('inf')
        converged = final_loss < 1000
        
        print(f"   Learning Rate = {lr:.5f} ({description}):")
        print(f"   Final Loss: {final_loss:,.0f}")
        print(f"   Converged: {'✅ Yes' if converged else '❌ No'}")
        print()
    
    print("💡 KEY INSIGHT:")
    print("   • Too small: Slow but steady progress")
    print("   • Just right: Fast convergence ✓")
    print("   • Too large: Overshoots, unstable")
    print()


# ============================================================================
# 6. Real-World Example: Salary Prediction
# ============================================================================

def demo_salary_prediction():
    """
    Real-world problem: Predict salary from years of experience.
    
    This is the kind of problem you'll actually encounter!
    """
    print("\n" + "=" * 70)
    print("5. Real-World Example: Salary Prediction")
    print("=" * 70)
    print()
    print("🎯 Problem: Predict salary from years of experience")
    print()
    
    # Dataset: years of experience -> salary ($1000s)
    experience = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    salary = [45, 50, 60, 65, 70, 80, 85, 90, 100, 105]  # In $1000s
    
    print("   Training Data (10 employees):")
    for exp, sal in zip(experience[:5], salary[:5]):
        print(f"   {exp} years → ${sal}k")
    print("   ...")
    print()
    
    # Train model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(experience, salary, verbose=False)
    
    print("✅ Model Trained!")
    print()
    print(f"   Equation: Salary = ${model.slope:.2f}k × Years + ${model.intercept:.2f}k")
    print()
    print("   Interpretation:")
    print(f"   • Base salary (0 years): ${model.intercept:.0f}k")
    print(f"   • Raise per year: ${model.slope:.2f}k")
    print()
    
    # Make predictions for new employees
    print("🔮 Predictions for New Hires:")
    print()
    
    new_experience = [3.5, 6.5, 12]
    predictions = model.predict(new_experience)
    
    for exp, pred in zip(new_experience, predictions):
        print(f"   {exp:.1f} years experience → ${pred:.0f}k predicted salary")
    print()
    
    # Model performance
    r2 = model.score(experience, salary)
    print(f"📈 Model Performance: R² = {r2:.4f}")
    print()
    
    if r2 > 0.9:
        print("   🎉 Excellent fit! Salary strongly correlated with experience.")
    elif r2 > 0.7:
        print("   ✅ Good fit. Experience explains most salary variation.")
    else:
        print("   ⚠️  Moderate fit. Other factors also matter (location, role, etc.)")
    print()


# ============================================================================
# 7. When Linear Regression Works (and When It Doesn't)
# ============================================================================

def demo_when_to_use():
    """
    Understanding when linear regression is appropriate.
    """
    print("\n" + "=" * 70)
    print("6. When to Use Linear Regression")
    print("=" * 70)
    print()
    
    print("✅ GOOD USE CASES:")
    print()
    print("   1. Linear Relationship:")
    print("      • Salary vs Experience")
    print("      • Sales vs Advertising Budget")
    print("      • Distance vs Time (constant speed)")
    print()
    
    print("   2. Continuous Output:")
    print("      • Predicting prices (not categories)")
    print("      • Forecasting temperatures")
    print("      • Estimating quantities")
    print()
    
    print("   3. Quick Baseline:")
    print("      • Fast to train")
    print("      • Easy to interpret")
    print("      • Good starting point before complex models")
    print()
    
    print("❌ BAD USE CASES:")
    print()
    print("   1. Non-Linear Relationships:")
    print("      • Age vs Medical Risk (U-shaped curve)")
    print("      • Temperature vs Ice Cream Sales (only above freezing)")
    print("      • Solution: Use polynomial features or non-linear models")
    print()
    
    print("   2. Classification Problems:")
    print("      • Spam vs Not Spam (yes/no, not a number)")
    print("      • Cat vs Dog (categories, not continuous)")
    print("      • Solution: Use Logistic Regression or other classifiers")
    print()
    
    print("   3. Multiple Complex Interactions:")
    print("      • House prices depend on size, location, age, condition...")
    print("      • Solution: Multiple Linear Regression (next level!)")
    print()
    
    print("💡 RULE OF THUMB:")
    print("   If you can draw a reasonably straight line through your data,")
    print("   linear regression will work well!")
    print()


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all linear regression demonstrations."""
    print("\n🤖 Linear Regression From Scratch\n")
    print("Learn by building! Understand every line of code.")
    print()
    
    demo_linear_regression_intuition()
    demo_cost_function()
    demo_training()
    demo_gradient_descent_intuition()
    demo_salary_prediction()
    demo_when_to_use()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Linear Regression: Find best straight line through data
   • Model: y = slope × x + intercept
   • Goal: Minimize prediction errors (MSE)

2. Gradient Descent: Optimization algorithm
   • Start with random parameters
   • Calculate error (loss)
   • Update parameters to reduce error
   • Repeat until convergence

3. Learning Rate: Controls step size
   • Too small: Slow convergence
   • Too large: Overshoots, unstable
   • Just right: Fast, stable convergence

4. Mean Squared Error (MSE): Loss function
   • Measures prediction quality
   • Penalizes big errors more than small ones
   • Goal: Find parameters that minimize MSE

5. R² Score: Model performance metric
   • 1.0 = Perfect predictions
   • 0.0 = As good as predicting the mean
   • < 0 = Worse than mean

6. When to Use:
   ✅ Linear relationships
   ✅ Continuous outputs
   ✅ Quick baseline model
   ❌ Non-linear patterns (use polynomial or other models)
   ❌ Classification (use logistic regression)

WHAT YOU BUILT:
• Complete linear regression from scratch
• Gradient descent optimizer
• Loss tracking and convergence monitoring
• Prediction and scoring functions

This is the FOUNDATION of machine learning!
Every complex model uses these same principles:
- Define a model
- Define a loss function
- Use gradient descent to optimize
- Make predictions

Next: 02_logistic_regression.py - Classification problems!
""")


if __name__ == "__main__":
    main()
