"""
Logistic Regression From Scratch

Build classification model from the ground up!
Focus: Understanding binary classification and the sigmoid function.

No libraries needed for core implementation
Run: poetry run python 06-machine-learning-fundamentals/from_scratch/02_logistic_regression.py
"""

import random
import math
from typing import List, Tuple


# ============================================================================
# 1. What is Logistic Regression? The Intuition
# ============================================================================

def demo_logistic_regression_intuition():
    """
    Logistic Regression: Predict YES/NO, not numbers.
    
    INTUITION - The Email Spam Filter:
    
    Linear Regression predicts numbers: "This house costs $300k"
    Logistic Regression predicts probabilities: "This email is 85% spam"
    
    The Problem with Linear Regression for Classification:
    - Linear: "This email is 1.5 spam" ← Makes no sense!
    - Logistic: "This email is 85% probability of spam" ← Perfect!
    
    THE MAGIC: Sigmoid Function
    
    Takes any number and squashes it to 0-1 range (probability):
    - Input: -infinity to +infinity
    - Output: 0 to 1 (perfect for probabilities!)
    
    Examples:
    - Very negative → ~0 (definitely NO)
    - Zero → 0.5 (unsure)
    - Very positive → ~1 (definitely YES)
    
    REAL-WORLD USE CASES:
    - Email: Spam or Not Spam?
    - Medicine: Disease or Healthy?
    - Finance: Fraud or Legitimate?
    - Marketing: Click or No Click?
    - Any YES/NO decision!
    
    KEY DIFFERENCE FROM LINEAR REGRESSION:
    - Linear: Predicts continuous values (prices, temperatures)
    - Logistic: Predicts probabilities (0 to 1, YES or NO)
    """
    print("=" * 70)
    print("1. Logistic Regression Intuition")
    print("=" * 70)
    print()
    print("💭 INTUITION: Email Spam Detection")
    print()
    print("   Linear Regression (WRONG for this!):")
    print("   • Input: Email with word 'FREE'")
    print("   • Output: 1.8 spam ← What does 1.8 spam mean?!")
    print()
    print("   Logistic Regression (RIGHT!):")
    print("   • Input: Email with word 'FREE'")
    print("   • Output: 0.85 = 85% chance it's spam ✓")
    print()
    print("🎯 THE KEY: Sigmoid Function")
    print()
    print("   Converts any number to probability (0-1):")
    print("   • -10 → 0.00 (definitely NO)")
    print("   •  -2 → 0.12 (probably NO)")
    print("   •   0 → 0.50 (unsure)")
    print("   •  +2 → 0.88 (probably YES)")
    print("   • +10 → 1.00 (definitely YES)")
    print()
    print("📊 Real-World Classification Problems:")
    print("   • Medical: Tumor is Malignant? (YES/NO)")
    print("   • Finance: Transaction is Fraud? (YES/NO)")
    print("   • Marketing: User will Click? (YES/NO)")
    print("   • HR: Candidate will Succeed? (YES/NO)")
    print()


# ============================================================================
# 2. The Sigmoid Function
# ============================================================================

def demo_sigmoid_function():
    """
    Sigmoid Function: The heart of logistic regression.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    Properties:
    - Output always between 0 and 1
    - Smooth S-shaped curve
    - Output = 0.5 when input = 0
    
    Why it works:
    - Large negative x → Output ~0
    - Large positive x → Output ~1
    - Perfect for probabilities!
    """
    print("\n" + "=" * 70)
    print("2. The Sigmoid Function")
    print("=" * 70)
    print()
    print("📐 Formula: σ(x) = 1 / (1 + e^(-x))")
    print()
    print("   Properties:")
    print("   • Input: Any number (-∞ to +∞)")
    print("   • Output: Always 0 to 1")
    print("   • S-shaped curve")
    print()
    
    def sigmoid(x: float) -> float:
        """Calculate sigmoid function."""
        return 1 / (1 + math.exp(-x))
    
    print("🔢 Examples:")
    print()
    
    test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    for x in test_values:
        sig = sigmoid(x)
        print(f"   σ({x:3d}) = {sig:.4f}", end="")
        
        if sig < 0.3:
            print("  ← Strong NO")
        elif sig < 0.4:
            print("  ← Probably NO")
        elif sig < 0.6:
            print("  ← Unsure")
        elif sig < 0.7:
            print("  ← Probably YES")
        else:
            print("  ← Strong YES")
    
    print()
    print("💡 INTERPRETATION:")
    print("   σ(x) < 0.5 → Predict class 0 (NO)")
    print("   σ(x) ≥ 0.5 → Predict class 1 (YES)")
    print()


# ============================================================================
# 3. Implementation: Logistic Regression Class
# ============================================================================

class LogisticRegression:
    """
    Logistic Regression from scratch using gradient descent.
    
    Model: P(y=1|x) = σ(w*x + b)
    where σ is the sigmoid function
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = 0.0
        self.bias = 0.0
        self.history = []
    
    @staticmethod
    def _sigmoid(z: float) -> float:
        """
        Sigmoid function: σ(z) = 1 / (1 + e^(-z))
        
        Handles overflow by clipping extreme values.
        """
        z = max(-500, min(500, z))  # Prevent overflow
        return 1 / (1 + math.exp(-z))
    
    def predict_proba(self, x: List[float]) -> List[float]:
        """
        Predict probabilities for each sample.
        
        Returns values between 0 and 1.
        """
        probabilities = []
        for xi in x:
            z = self.weights * xi + self.bias
            prob = self._sigmoid(z)
            probabilities.append(prob)
        return probabilities
    
    def predict(self, x: List[float], threshold: float = 0.5) -> List[int]:
        """
        Predict class labels (0 or 1).
        
        Args:
            x: Input features
            threshold: Decision boundary (default 0.5)
        
        Returns:
            List of 0s and 1s
        """
        probabilities = self.predict_proba(x)
        return [1 if p >= threshold else 0 for p in probabilities]
    
    def _calculate_loss(self, x: List[float], y: List[int]) -> float:
        """
        Calculate Binary Cross-Entropy loss.
        
        Formula: -average(y*log(p) + (1-y)*log(1-p))
        
        This measures how far our predictions are from true labels.
        """
        probabilities = self.predict_proba(x)
        n = len(y)
        
        loss = 0
        for i in range(n):
            p = probabilities[i]
            # Avoid log(0) by clipping
            p = max(1e-15, min(1 - 1e-15, p))
            
            if y[i] == 1:
                loss -= math.log(p)
            else:
                loss -= math.log(1 - p)
        
        return loss / n
    
    def _calculate_gradients(self, x: List[float], y: List[int]) -> Tuple[float, float]:
        """
        Calculate gradients for weights and bias.
        
        Gradient = average((prediction - actual) * feature)
        """
        n = len(x)
        probabilities = self.predict_proba(x)
        
        # Calculate errors
        errors = [probabilities[i] - y[i] for i in range(n)]
        
        # Gradients
        weight_gradient = sum(errors[i] * x[i] for i in range(n)) / n
        bias_gradient = sum(errors) / n
        
        return weight_gradient, bias_gradient
    
    def fit(self, x: List[float], y: List[int], verbose: bool = False) -> None:
        """
        Train the model using gradient descent.
        """
        # Initialize parameters
        self.weights = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)
        
        for iteration in range(self.iterations):
            # Calculate loss
            loss = self._calculate_loss(x, y)
            self.history.append(loss)
            
            # Calculate gradients
            weight_grad, bias_grad = self._calculate_gradients(x, y)
            
            # Update parameters
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad
            
            # Print progress
            if verbose and (iteration % 100 == 0 or iteration == self.iterations - 1):
                accuracy = self.score(x, y)
                print(f"   Iteration {iteration:4d}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")
    
    def score(self, x: List[float], y: List[int]) -> float:
        """
        Calculate accuracy: percentage of correct predictions.
        """
        predictions = self.predict(x)
        correct = sum(1 for i in range(len(y)) if predictions[i] == y[i])
        return correct / len(y)


# ============================================================================
# 4. Demo: Training on Simple Data
# ============================================================================

def demo_training():
    """
    Train logistic regression on exam pass/fail data.
    """
    print("\n" + "=" * 70)
    print("3. Training Logistic Regression")
    print("=" * 70)
    print()
    print("📊 Problem: Predict Exam Pass/Fail from Study Hours")
    print()
    
    # Dataset: study hours -> pass (1) or fail (0)
    study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    passed = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0=fail, 1=pass
    
    print("   Training Data:")
    for hours, result in zip(study_hours, passed):
        status = "Pass ✓" if result == 1 else "Fail ✗"
        print(f"   {hours:2d} hours → {status}")
    print()
    
    # Train model
    print("🚀 Training Model...")
    print()
    
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(study_hours, passed, verbose=True)
    
    print()
    print("✅ Training Complete!")
    print()
    
    # Make predictions
    print("🔮 Predictions:")
    print()
    
    test_hours = [2.5, 5, 7.5]
    probabilities = model.predict_proba(test_hours)
    predictions = model.predict(test_hours)
    
    for hours, prob, pred in zip(test_hours, probabilities, predictions):
        result = "Pass ✓" if pred == 1 else "Fail ✗"
        print(f"   {hours:.1f} hours → {prob:.2%} pass probability → {result}")
    print()
    
    # Model performance
    accuracy = model.score(study_hours, passed)
    print(f"📈 Training Accuracy: {accuracy:.1%}")
    print()


# ============================================================================
# 5. Real-World Example: Medical Diagnosis
# ============================================================================

def demo_medical_diagnosis():
    """
    Real-world: Predict disease from symptom severity.
    """
    print("\n" + "=" * 70)
    print("4. Real-World: Medical Diagnosis")
    print("=" * 70)
    print()
    print("🏥 Problem: Predict Disease from Symptom Severity (0-10)")
    print()
    
    # Dataset: symptom severity -> disease (1) or healthy (0)
    severity = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10]
    disease = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    print("   Training Data (14 patients):")
    for sev, dis in zip(severity[:7], disease[:7]):
        status = "Disease 🔴" if dis == 1 else "Healthy ✓"
        print(f"   Severity {sev:2d}/10 → {status}")
    print("   ...")
    print()
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(severity, disease, verbose=False)
    
    print("✅ Model Trained!")
    print()
    
    # Analyze decision boundary
    print("🎯 Decision Boundary Analysis:")
    print()
    
    # Find where probability = 0.5
    # At decision boundary: weights * x + bias = 0
    boundary = -model.bias / model.weights if model.weights != 0 else 0
    print(f"   Decision boundary at severity: {boundary:.1f}/10")
    print()
    print("   Interpretation:")
    print(f"   • Below {boundary:.1f}: Predict Healthy")
    print(f"   • Above {boundary:.1f}: Predict Disease")
    print()
    
    # Make predictions for new patients
    print("🔮 Diagnosing New Patients:")
    print()
    
    new_patients = [3, 5, 7, 9]
    probs = model.predict_proba(new_patients)
    preds = model.predict(new_patients)
    
    for sev, prob, pred in zip(new_patients, probs, preds):
        diagnosis = "Disease 🔴" if pred == 1 else "Healthy ✓"
        confidence = prob if pred == 1 else (1 - prob)
        print(f"   Patient with severity {sev}/10:")
        print(f"   → {diagnosis} ({confidence:.1%} confidence)")
        print()
    
    # Performance
    accuracy = model.score(severity, disease)
    print(f"📈 Model Accuracy: {accuracy:.1%}")
    print()


# ============================================================================
# 6. Understanding the Decision Boundary
# ============================================================================

def demo_decision_boundary():
    """
    Visualize how logistic regression makes decisions.
    """
    print("\n" + "=" * 70)
    print("5. Understanding Decision Boundary")
    print("=" * 70)
    print()
    print("💭 INTUITION: Where Does the Model Draw the Line?")
    print()
    print("   Logistic Regression finds a threshold:")
    print("   • Below threshold → Predict class 0")
    print("   • Above threshold → Predict class 1")
    print()
    
    # Simple example
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    
    model = LogisticRegression(learning_rate=0.1, iterations=500)
    model.fit(x, y, verbose=False)
    
    # Find decision boundary
    boundary = -model.bias / model.weights if model.weights != 0 else 0
    
    print(f"📊 Example: Learned Decision Boundary = {boundary:.2f}")
    print()
    print("   Visual Representation:")
    print()
    print("   Input:  1   2   3   4 | 5   6   7   8")
    print("   Class:  0   0   0   0 | 1   1   1   1")
    print(f"                         ↑ Boundary at {boundary:.1f}")
    print()
    
    # Show probabilities around boundary
    print("   Probabilities near boundary:")
    test_points = [boundary - 1, boundary - 0.5, boundary, boundary + 0.5, boundary + 1]
    probs = model.predict_proba(test_points)
    
    for point, prob in zip(test_points, probs):
        print(f"   x={point:.2f} → P(y=1)={prob:.2%}", end="")
        if prob < 0.5:
            print("  → Predict 0")
        else:
            print("  → Predict 1")
    print()


# ============================================================================
# 7. When to Use Logistic Regression
# ============================================================================

def demo_when_to_use():
    """
    Understanding when logistic regression is appropriate.
    """
    print("\n" + "=" * 70)
    print("6. When to Use Logistic Regression")
    print("=" * 70)
    print()
    
    print("✅ PERFECT FOR:")
    print()
    print("   1. Binary Classification (YES/NO):")
    print("      • Spam detection (spam/not spam)")
    print("      • Medical diagnosis (disease/healthy)")
    print("      • Fraud detection (fraud/legitimate)")
    print("      • Churn prediction (will leave/will stay)")
    print()
    
    print("   2. Need Probabilities:")
    print("      • Not just prediction, but confidence")
    print("      • 'This email is 95% spam' better than just 'spam'")
    print("      • Helps with threshold tuning")
    print()
    
    print("   3. Interpretable Model:")
    print("      • Simple to explain to stakeholders")
    print("      • Feature weights show importance")
    print("      • Fast training and prediction")
    print()
    
    print("❌ NOT GOOD FOR:")
    print()
    print("   1. Multi-Class Problems (>2 classes):")
    print("      • Cat/Dog/Bird classification")
    print("      • Solution: Softmax regression or one-vs-all")
    print()
    
    print("   2. Non-Linear Decision Boundaries:")
    print("      • When classes overlap in complex ways")
    print("      • Solution: Add polynomial features or use non-linear models")
    print()
    
    print("   3. Regression Problems:")
    print("      • Predicting house prices (continuous values)")
    print("      • Solution: Use Linear Regression instead")
    print()
    
    print("💡 QUICK DECISION:")
    print("   Binary outcome (0/1, YES/NO) → Logistic Regression")
    print("   Continuous outcome (numbers) → Linear Regression")
    print()


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all logistic regression demonstrations."""
    print("\n🎯 Logistic Regression From Scratch\n")
    print("Classification made simple!")
    print()
    
    demo_logistic_regression_intuition()
    demo_sigmoid_function()
    demo_training()
    demo_medical_diagnosis()
    demo_decision_boundary()
    demo_when_to_use()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Logistic Regression: Binary classification (YES/NO)
   • Predicts probabilities (0 to 1)
   • Uses sigmoid function to squash outputs
   • Decision boundary: probability = 0.5

2. Sigmoid Function: σ(x) = 1 / (1 + e^(-x))
   • Converts any number to probability
   • S-shaped curve
   • Output always between 0 and 1

3. Binary Cross-Entropy Loss:
   • Measures how far predictions are from true labels
   • Minimized using gradient descent
   • Penalizes confident wrong predictions more

4. Decision Boundary:
   • Threshold where model switches predictions
   • Usually at probability = 0.5
   • Can be tuned based on business needs

5. Interpretation:
   • Output = probability of class 1
   • < 0.5 → Predict class 0
   • ≥ 0.5 → Predict class 1

6. Use Cases:
   ✅ Email spam detection
   ✅ Medical diagnosis
   ✅ Fraud detection
   ✅ Click prediction
   ✅ Any binary YES/NO decision

7. Comparison:
   • Linear Regression: Predicts numbers
   • Logistic Regression: Predicts probabilities/classes

WHAT YOU BUILT:
• Complete logistic regression from scratch
• Sigmoid activation function
• Binary cross-entropy loss
• Gradient descent optimizer
• Probability and class predictions

This is the foundation of neural networks!
The sigmoid function is an activation function.
The same principles apply to deep learning.

Next: 03_decision_tree.py - Non-linear classification!
""")


if __name__ == "__main__":
    main()
