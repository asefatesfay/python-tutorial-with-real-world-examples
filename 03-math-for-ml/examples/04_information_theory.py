"""
Information Theory for ML/AI Engineering

Build intuition for entropy, cross-entropy, and loss functions.
Focus: Understanding why we use specific loss functions in ML.

Install: poetry add numpy matplotlib
Run: poetry run python 03-math-for-ml/examples/04_information_theory.py
"""

import math
from typing import List


# ============================================================================
# 1. Entropy - Measuring Uncertainty
# ============================================================================

def demo_entropy():
    """
    Entropy: Measure of uncertainty/surprise in data.
    
    Formula: H(X) = -Σ p(x) * log₂(p(x))
    
    INTUITION - Guessing Your Friend's Coffee Order:
    
    Scenario 1: Friend ALWAYS orders "Americano"
    - Entropy: 0 (zero surprise, you know what's coming)
    - Predictability: 100%
    - Information gained when they order: None (you already knew!)
    
    Scenario 2: Friend orders Americano 90%, Latte 10%
    - Entropy: Low (mostly predictable)
    - Surprise: Occasional ("Oh, latte today!")
    - You can usually guess right
    
    Scenario 3: Friend randomly picks from 10 drinks equally
    - Entropy: HIGH (maximum surprise!)
    - Predictability: 10% (just guessing)
    - You never know what's coming
    
    THE INSIGHT:
    Entropy = How many yes/no questions to guess the answer
    - 2 equally likely options (fair coin)? 1 question (1 bit)
    - 4 equally likely options? 2 questions (2 bits)
    - 8 equally likely options? 3 questions (3 bits)
    - 100 with one at 99%? Almost 0 questions (you know it!)
    
    In ML:
    - High entropy data = Hard to predict (need complex model)
    - Low entropy data = Easy to predict (simple model works)
    - Decision trees: Split to REDUCE entropy (make data more predictable)
    - Model confident = Low entropy output
    - Model uncertain = High entropy output
    
    Intuition:
    - High entropy: Unpredictable (fair coin)
    - Low entropy: Predictable (loaded coin)
    - Zero entropy: Certain (always heads)
    
    ML Use:
    - Decision trees (information gain)
    - Measure data distribution
    - Understanding loss functions
    """
    print("=" * 70)
    print("1. Entropy - Measuring Uncertainty")
    print("=" * 70)
    print()
    print("💭 INTUITION: Guessing Your Friend's Coffee Order")
    print()
    print("   Scenario 1: Always orders Americano")
    print("   → Entropy: 0 (No surprise, 100% predictable)")
    print()
    print("   Scenario 2: 90% Americano, 10% Latte")
    print("   → Entropy: Low (Mostly predictable)")
    print()
    print("   Scenario 3: Randomly picks from 10 drinks")
    print("   → Entropy: HIGH (Maximum surprise!)")
    print()
    print("   Higher entropy = More unpredictable = Harder to guess")
    print()
    print("   In ML:")
    print("   - High entropy data = Need complex model")
    print("   - Low entropy data = Simple model works fine")
    print("   - Decision trees split to REDUCE entropy")
    print()
    
    def entropy(probabilities: List[float]) -> float:
        """
        Calculate entropy.
        
        H(X) = -Σ p(x) * log₂(p(x))
        
        Higher entropy = more uncertainty
        """
        H = 0
        for p in probabilities:
            if p > 0:  # Avoid log(0)
                H -= p * math.log2(p)
        return H
    
    print("🎲 Example 1: Coin Flips")
    print()
    
    # Fair coin
    fair_coin = [0.5, 0.5]  # P(H) = 0.5, P(T) = 0.5
    H_fair = entropy(fair_coin)
    
    print(f"Fair coin: P(H) = {fair_coin[0]}, P(T) = {fair_coin[1]}")
    print(f"  Entropy: {H_fair:.3f} bits")
    print(f"  Interpretation: Maximum uncertainty (can't predict)")
    print()
    
    # Loaded coin
    loaded_coin = [0.9, 0.1]  # P(H) = 0.9, P(T) = 0.1
    H_loaded = entropy(loaded_coin)
    
    print(f"Loaded coin: P(H) = {loaded_coin[0]}, P(T) = {loaded_coin[1]}")
    print(f"  Entropy: {H_loaded:.3f} bits")
    print(f"  Interpretation: Less uncertainty (mostly heads)")
    print()
    
    # Certain
    certain = [1.0, 0.0]  # P(H) = 1.0, P(T) = 0.0
    H_certain = entropy(certain)
    
    print(f"Certain: P(H) = {certain[0]}, P(T) = {certain[1]}")
    print(f"  Entropy: {H_certain:.3f} bits")
    print(f"  Interpretation: No uncertainty (always heads)")
    print()
    
    # More outcomes
    print("🎲 Example 2: 6-sided Die")
    print()
    
    # Fair die
    fair_die = [1/6] * 6
    H_fair_die = entropy(fair_die)
    
    print(f"Fair die: P(each) = {fair_die[0]:.3f}")
    print(f"  Entropy: {H_fair_die:.3f} bits")
    print(f"  Interpretation: High uncertainty (6 equal outcomes)")
    print()
    
    # Loaded die
    loaded_die = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    H_loaded_die = entropy(loaded_die)
    
    print(f"Loaded die: {loaded_die}")
    print(f"  Entropy: {H_loaded_die:.3f} bits")
    print(f"  Interpretation: Lower uncertainty (mostly 1)")
    print()
    
    print("💡 Entropy increases with:")
    print("   - More possible outcomes")
    print("   - More uniform distribution")
    print()
    print("In ML:")
    print("   - Decision trees: Split to minimize entropy")
    print("   - Information gain = Entropy before - Entropy after")
    print("   - High entropy data: Harder to predict")


# ============================================================================
# 2. Cross-Entropy - Comparing Distributions
# ============================================================================

def demo_cross_entropy():
    """
    Cross-Entropy: Measure difference between two distributions.
    
    Formula: H(P, Q) = -Σ p(x) * log(q(x))
    - P: True distribution
    - Q: Predicted distribution
    
    INTUITION - Weather Forecast Accuracy:
    
    Tomorrow it WILL rain (truth = 100% rain, 0% sun).
    
    Forecast 1: "100% rain" → Perfect! Cross-entropy = 0
    - You trust it, bring umbrella ✓
    
    Forecast 2: "90% rain, 10% sun" → Pretty good, cross-entropy = 0.15
    - Mostly right, you still bring umbrella
    
    Forecast 3: "50% rain, 50% sun" → Very uncertain! Cross-entropy = 1.0
    - Useless forecast, you don't know what to do
    
    Forecast 4: "10% rain, 90% sun" → WRONG and confident! Cross-entropy = 3.3
    - You leave umbrella, get soaked!
    - This is THE WORST - wrong with high confidence
    
    THE KEY INSIGHT:
    Cross-entropy penalizes CONFIDENT MISTAKES more than uncertainty:
    - Uncertain but right direction: Medium penalty
    - Uncertain wrong direction: Medium penalty  
    - Confident and RIGHT: Small penalty ✓
    - Confident and WRONG: HUGE penalty! ✗✗✗
    
    This is why it's perfect for ML:
    - Model says "100% cat" but it's a dog? MASSIVE loss (learn a lot!)
    - Model says "33% cat, 33% dog, 33% bird" but it's a dog? Medium loss
    - Forces model to be confident AND correct
    
    In Classification:
    - True label: [1, 0, 0] (it's a cat)
    - Good prediction: [0.9, 0.05, 0.05] → Low cross-entropy ✓
    - Bad prediction: [0.1, 0.8, 0.1] → High cross-entropy ✗
    - Training minimizes cross-entropy → Model learns to be right!
    
    ML Use:
    - Loss function for classification
    - Measures how well model predicts
    - Lower is better
    """
    print("\n" + "=" * 70)
    print("2. Cross-Entropy - Comparing Distributions")
    print("=" * 70)
    print()
    print("💭 INTUITION: Weather Forecast Accuracy")
    print()
    print("   Tomorrow it WILL rain (truth: 100% rain).")
    print()
    print("   Forecast 1: '100% rain'")
    print("   → Cross-entropy: 0 (Perfect!) ✓")
    print()
    print("   Forecast 2: '90% rain, 10% sun'")
    print("   → Cross-entropy: 0.15 (Pretty good)")
    print()
    print("   Forecast 3: '50% rain, 50% sun'")
    print("   → Cross-entropy: 1.0 (Useless, uncertain)")
    print()
    print("   Forecast 4: '10% rain, 90% sun'")
    print("   → Cross-entropy: 3.3 (WRONG & confident = WORST!) ✗")
    print()
    print("   Key: Cross-entropy penalizes confident mistakes heavily!")
    print()
    print("   In ML: Model confident but wrong? Huge loss = Learn more!")
    print("   This is why cross-entropy is the standard classification loss.")
    print()
    
    def cross_entropy(true_probs: List[float], pred_probs: List[float]) -> float:
        """
        Calculate cross-entropy between true and predicted distributions.
        
        H(P, Q) = -Σ p(x) * log(q(x))
        """
        ce = 0
        for p, q in zip(true_probs, pred_probs):
            if p > 0 and q > 0:
                ce -= p * math.log(q)
        return ce
    
    print("🎯 Example: Image Classification (Cat, Dog, Bird)")
    print()
    
    # True label: Cat (one-hot encoded)
    true_label = [1.0, 0.0, 0.0]  # [Cat, Dog, Bird]
    
    print(f"True label: Cat → {true_label}")
    print()
    
    # Good prediction
    good_pred = [0.9, 0.05, 0.05]
    ce_good = cross_entropy(true_label, good_pred)
    
    print(f"Good prediction: {good_pred}")
    print(f"  Cross-entropy: {ce_good:.3f}")
    print(f"  Model is confident and correct! ✅")
    print()
    
    # Medium prediction
    medium_pred = [0.6, 0.3, 0.1]
    ce_medium = cross_entropy(true_label, medium_pred)
    
    print(f"Medium prediction: {good_pred}")
    print(f"  Cross-entropy: {ce_medium:.3f}")
    print(f"  Model is somewhat confident ⚠️")
    print()
    
    # Bad prediction
    bad_pred = [0.1, 0.8, 0.1]
    ce_bad = cross_entropy(true_label, bad_pred)
    
    print(f"Bad prediction: {bad_pred}")
    print(f"  Cross-entropy: {ce_bad:.3f}")
    print(f"  Model is wrong and confident! ❌")
    print()
    
    # Uncertain prediction
    uncertain_pred = [0.33, 0.33, 0.34]
    ce_uncertain = cross_entropy(true_label, uncertain_pred)
    
    print(f"Uncertain prediction: {uncertain_pred}")
    print(f"  Cross-entropy: {ce_uncertain:.3f}")
    print(f"  Model is unsure (better than being wrong!)")
    print()
    
    print("💡 Key insights:")
    print(f"   - Lower cross-entropy = better prediction")
    print(f"   - Perfect prediction (1.0 for correct class): {cross_entropy([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]):.3f}")
    print(f"   - Wrong and confident: High loss (model needs correction!)")
    print()
    print("In ML:")
    print("   - Cross-entropy loss for classification")
    print("   - Binary cross-entropy for binary classification")
    print("   - Categorical cross-entropy for multi-class")


# ============================================================================
# 3. KL Divergence - Distribution Distance
# ============================================================================

def demo_kl_divergence():
    """
    KL Divergence: Measure "distance" between distributions.
    
    Formula: D_KL(P||Q) = Σ p(x) * log(p(x) / q(x))
    
    Properties:
    - Always >= 0
    - = 0 only if P = Q
    - Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
    
    ML Use:
    - Variational autoencoders (VAE)
    - Regularization term
    - Comparing model distributions
    """
    print("\n" + "=" * 70)
    print("3. KL Divergence - Distribution Distance")
    print("=" * 70)
    
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """
        Calculate KL divergence from Q to P.
        
        D_KL(P||Q) = Σ p(x) * log(p(x) / q(x))
        """
        kl = 0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)
        return kl
    
    print("🎯 Example: Modeling User Preferences")
    print()
    
    # True user preference (from behavior data)
    true_preference = [0.5, 0.3, 0.2]  # [Action, Comedy, Drama]
    
    print(f"True preference: {true_preference}")
    print(f"                 [Action, Comedy, Drama]")
    print()
    
    # Model 1: Close to true
    model1 = [0.45, 0.35, 0.2]
    kl1 = kl_divergence(true_preference, model1)
    
    print(f"Model 1: {model1}")
    print(f"  KL Divergence: {kl1:.4f}")
    print(f"  Close to true distribution ✅")
    print()
    
    # Model 2: Somewhat off
    model2 = [0.4, 0.4, 0.2]
    kl2 = kl_divergence(true_preference, model2)
    
    print(f"Model 2: {model2}")
    print(f"  KL Divergence: {kl2:.4f}")
    print(f"  Somewhat different ⚠️")
    print()
    
    # Model 3: Very different
    model3 = [0.2, 0.2, 0.6]
    kl3 = kl_divergence(true_preference, model3)
    
    print(f"Model 3: {model3}")
    print(f"  KL Divergence: {kl3:.4f}")
    print(f"  Very different! ❌")
    print()
    
    # Uniform (uninformative)
    uniform = [1/3, 1/3, 1/3]
    kl_uniform = kl_divergence(true_preference, uniform)
    
    print(f"Uniform: {uniform}")
    print(f"  KL Divergence: {kl_uniform:.4f}")
    print(f"  Maximum uncertainty")
    print()
    
    print("💡 Relation to Cross-Entropy:")
    print("   H(P, Q) = H(P) + D_KL(P||Q)")
    print("   Cross-Entropy = Entropy + KL Divergence")
    print()
    print("In ML:")
    print("   - VAE: Regularize latent space")
    print("   - RL: Policy gradient methods")
    print("   - Model distillation: Match teacher distribution")


# ============================================================================
# 4. Loss Functions in ML
# ============================================================================

def demo_loss_functions():
    """
    Loss Functions: Measure how wrong model predictions are.
    
    Common losses:
    - MSE: Regression problems
    - Cross-Entropy: Classification problems
    - Binary Cross-Entropy: Binary classification
    - Hinge Loss: SVM, margin-based learning
    """
    print("\n" + "=" * 70)
    print("4. Loss Functions in ML")
    print("=" * 70)
    
    # Mean Squared Error (MSE)
    def mse(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Squared Error for regression."""
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    # Mean Absolute Error (MAE)
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Error for regression."""
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
    
    # Binary Cross-Entropy
    def binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float:
        """Binary cross-entropy for binary classification."""
        bce = 0
        for yt, yp in zip(y_true, y_pred):
            # Clip to avoid log(0)
            yp = max(min(yp, 0.9999), 0.0001)
            bce -= yt * math.log(yp) + (1 - yt) * math.log(1 - yp)
        return bce / len(y_true)
    
    print("📊 Regression: Predicting House Prices")
    print()
    
    # True prices (in $100k)
    true_prices = [3.0, 4.5, 5.0, 6.2]
    
    # Predictions
    pred_good = [3.1, 4.4, 5.1, 6.0]
    pred_bad = [2.0, 5.5, 4.0, 7.0]
    
    mse_good = mse(true_prices, pred_good)
    mse_bad = mse(true_prices, pred_bad)
    
    mae_good = mae(true_prices, pred_good)
    mae_bad = mae(true_prices, pred_bad)
    
    print(f"True: {true_prices}")
    print()
    print(f"Good predictions: {pred_good}")
    print(f"  MSE: {mse_good:.3f}")
    print(f"  MAE: {mae_good:.3f}")
    print()
    print(f"Bad predictions: {pred_bad}")
    print(f"  MSE: {mse_bad:.3f}")
    print(f"  MAE: {mae_bad:.3f}")
    print()
    
    print("MSE vs MAE:")
    print("  MSE: Penalizes large errors more (squares them)")
    print("  MAE: Treats all errors equally")
    print("  Use MSE when outliers are important")
    print()
    
    print("🎯 Binary Classification: Spam Detection")
    print()
    
    # True labels (0 = not spam, 1 = spam)
    true_labels = [1, 0, 1, 0, 1]
    
    # Good model (confident and correct)
    pred_good = [0.9, 0.1, 0.85, 0.15, 0.95]
    bce_good = binary_cross_entropy(true_labels, pred_good)
    
    print(f"True: {true_labels} (1=spam, 0=not spam)")
    print()
    print(f"Good model: {pred_good}")
    print(f"  BCE: {bce_good:.3f} ✅")
    print()
    
    # Bad model (uncertain or wrong)
    pred_bad = [0.5, 0.5, 0.4, 0.6, 0.5]
    bce_bad = binary_cross_entropy(true_labels, pred_bad)
    
    print(f"Bad model: {pred_bad}")
    print(f"  BCE: {bce_bad:.3f} ❌")
    print()
    
    print("💡 Choosing loss function:")
    print("   - Regression: MSE, MAE, Huber")
    print("   - Binary classification: Binary cross-entropy")
    print("   - Multi-class: Categorical cross-entropy")
    print("   - Object detection: Combination of losses")


# ============================================================================
# 5. Real-World: Training with Different Losses
# ============================================================================

def demo_training_with_losses():
    """
    Real-world: Train same model with different loss functions.
    
    Shows how loss function choice affects learning.
    """
    print("\n" + "=" * 70)
    print("5. Real-World: Training with Different Losses")
    print("=" * 70)
    
    print("📊 Problem: Predict if student passes (>= 60%)")
    print()
    
    # Data: [study_hours, pass/fail]
    data = [
        (1, 0), (2, 0), (3, 0),  # Low hours → Fail
        (4, 0), (5, 1),          # Borderline
        (6, 1), (7, 1), (8, 1)   # High hours → Pass
    ]
    
    print("Training data (hours, pass):")
    for hours, passed in data:
        result = "Pass" if passed else "Fail"
        print(f"  {hours}h → {result}")
    print()
    
    def sigmoid(x: float) -> float:
        """Sigmoid activation."""
        return 1 / (1 + math.exp(-x))
    
    def train_model(loss_type: str, epochs: int = 100):
        """Train simple logistic regression with different losses."""
        # Model: P(pass) = sigmoid(w * hours + b)
        w = 0.0
        b = 0.0
        lr = 0.1
        
        for epoch in range(epochs):
            total_loss = 0
            dw = 0
            db = 0
            
            for hours, y_true in data:
                # Forward pass
                z = w * hours + b
                y_pred = sigmoid(z)
                
                # Loss and gradients
                if loss_type == "BCE":
                    # Binary cross-entropy
                    loss = -(y_true * math.log(y_pred + 1e-10) + 
                            (1 - y_true) * math.log(1 - y_pred + 1e-10))
                    # Gradient (from calculus)
                    error = y_pred - y_true
                else:  # MSE
                    # Mean squared error
                    loss = (y_true - y_pred) ** 2
                    # Gradient
                    error = -(y_true - y_pred) * y_pred * (1 - y_pred)
                
                total_loss += loss
                dw += error * hours
                db += error
            
            # Update weights
            w -= lr * dw / len(data)
            b -= lr * db / len(data)
        
        return w, b, total_loss / len(data)
    
    # Train with BCE
    print("Training with Binary Cross-Entropy:")
    w_bce, b_bce, loss_bce = train_model("BCE")
    print(f"  Final: w = {w_bce:.3f}, b = {b_bce:.3f}, loss = {loss_bce:.3f}")
    
    # Train with MSE
    print("\nTraining with Mean Squared Error:")
    w_mse, b_mse, loss_mse = train_model("MSE")
    print(f"  Final: w = {w_mse:.3f}, b = {b_mse:.3f}, loss = {loss_mse:.3f}")
    
    print()
    print("Test predictions:")
    test_hours = [2.5, 5, 7.5]
    
    print("\nBCE Model:")
    for h in test_hours:
        pred = sigmoid(w_bce * h + b_bce)
        result = "Pass" if pred >= 0.5 else "Fail"
        print(f"  {h}h → P(pass) = {pred:.3f} → {result}")
    
    print("\nMSE Model:")
    for h in test_hours:
        pred = sigmoid(w_mse * h + b_mse)
        result = "Pass" if pred >= 0.5 else "Fail"
        print(f"  {h}h → P(pass) = {pred:.3f} → {result}")
    
    print()
    print("💡 Observation:")
    print("   - BCE: Better for classification (probabilities)")
    print("   - MSE: Works but less principled for probabilities")
    print("   - Loss function should match problem type!")


# ============================================================================
# 6. Practical Guidelines
# ============================================================================

def demo_practical_guidelines():
    """
    Practical guidelines for choosing metrics and losses.
    """
    print("\n" + "=" * 70)
    print("6. Practical Guidelines")
    print("=" * 70)
    
    print("📋 Loss Function Selection Guide:")
    print()
    
    guidelines = [
        ("Regression (continuous output)", [
            "MSE: Standard choice, penalizes outliers",
            "MAE: Robust to outliers",
            "Huber: Combines MSE and MAE benefits",
            "RMSE: Same as MSE, interpretable units"
        ]),
        ("Binary Classification", [
            "Binary Cross-Entropy: Standard choice",
            "Hinge Loss: For SVM-style margin",
            "Focal Loss: For class imbalance"
        ]),
        ("Multi-Class Classification", [
            "Categorical Cross-Entropy: Standard choice",
            "Sparse Cross-Entropy: When labels are integers",
            "Focal Loss: For hard examples"
        ]),
        ("Sequence Tasks", [
            "CTC Loss: Speech recognition",
            "Cross-Entropy: Language modeling",
            "BLEU/ROUGE: Machine translation"
        ])
    ]
    
    for task, losses in guidelines:
        print(f"• {task}:")
        for loss in losses:
            print(f"    - {loss}")
        print()
    
    print("⚡ Tips:")
    print()
    print("  1. Match loss to problem type")
    print("     Regression → MSE/MAE")
    print("     Classification → Cross-Entropy")
    print()
    print("  2. Consider data properties")
    print("     Outliers present → MAE or Huber")
    print("     Class imbalance → Weighted loss or Focal")
    print()
    print("  3. Monitor multiple metrics")
    print("     Optimize: Training loss")
    print("     Evaluate: Task-specific metrics (accuracy, F1, etc.)")
    print()
    print("  4. Use appropriate final activation")
    print("     Regression: None or linear")
    print("     Binary: Sigmoid")
    print("     Multi-class: Softmax")
    print()
    print("  5. Be aware of numerical stability")
    print("     Clip predictions before log")
    print("     Use stable implementations (PyTorch, TensorFlow)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n📊 Information Theory for ML/AI Engineering\n")
    print("Focus: Understanding entropy, cross-entropy, and loss functions!")
    print()
    
    demo_entropy()
    demo_cross_entropy()
    demo_kl_divergence()
    demo_loss_functions()
    demo_training_with_losses()
    demo_practical_guidelines()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Entropy: Measure of uncertainty
   - High: Unpredictable (fair coin)
   - Low: Predictable (loaded coin)
   - Zero: Certain (always same)

2. Cross-Entropy: Compare distributions
   - Measures prediction quality
   - Lower is better
   - Standard loss for classification

3. KL Divergence: Distribution "distance"
   - Always >= 0
   - = 0 only if distributions identical
   - Used in VAE, regularization

4. Loss Functions:
   - Regression: MSE, MAE
   - Binary: Binary Cross-Entropy
   - Multi-class: Categorical Cross-Entropy
   - Choose based on problem type!

5. Information Theory in ML:
   - Decision trees: Entropy, information gain
   - Classification: Cross-entropy loss
   - VAE: KL divergence regularization
   - RL: Policy entropy

Practical Tips:
- Match loss to problem type
- Consider data properties (outliers, imbalance)
- Use stable implementations
- Monitor multiple metrics

🎉 Congratulations! You've completed Math for ML/AI!
    
Next Steps:
- Module 4: NumPy & Pandas (efficient implementations)
- Module 5: Feature Engineering (apply statistics)
- Module 6: Neural Networks (put it all together!)
""")


if __name__ == "__main__":
    main()
