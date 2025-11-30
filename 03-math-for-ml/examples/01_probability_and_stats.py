"""
Probability and Statistics for ML/AI Engineering

Build intuition for statistics concepts you'll use in ML/AI.
Focus: Understanding WHY and WHEN to use these concepts, not memorizing formulas.

Install: poetry add numpy matplotlib scipy
Run: poetry run python 03-math-for-ml/examples/01_probability_and_stats.py
"""

import random
from collections import Counter
from typing import List


# ============================================================================
# 1. Mean, Variance, and Standard Deviation
# ============================================================================

def demo_mean_variance():
    """
    Mean: Average value (center of data)
    Variance: How spread out the data is
    Std Dev: Variance in same units as data
    
    INTUITION:
    Think of your daily commute time:
    - Mean: Your typical commute (e.g., 30 minutes)
    - Variance: How unpredictable it is
      * Low variance: Always 28-32 min (reliable, plan around it)
      * High variance: 15-60 min (unreliable, leave early or be late!)
    
    In ML: Two models with same average accuracy but different variance?
    Pick the consistent one! Low variance = predictable = production-ready.
    
    ML Use Cases:
    - Feature normalization
    - Outlier detection
    - Understanding model predictions
    """
    print("=" * 70)
    print("1. Mean, Variance, Standard Deviation")
    print("=" * 70)
    print()
    print("💭 INTUITION: Think of your daily commute")
    print("   Mean = Average time (your typical commute)")
    print("   Variance = How predictable it is")
    print("   • Low variance: 28-32 min → Plan your day confidently")
    print("   • High variance: 15-60 min → Always stressed about timing!")
    print()
    
    # Two datasets with same mean, different spread
    consistent_scores = [85, 86, 84, 85, 85, 86, 84, 85]
    variable_scores = [60, 95, 70, 100, 75, 90, 65, 85]
    
    def calculate_mean(data: List[float]) -> float:
        """Calculate average."""
        return sum(data) / len(data)
    
    def calculate_variance(data: List[float]) -> float:
        """Calculate variance: average of squared differences from mean."""
        mean = calculate_mean(data)
        squared_diffs = [(x - mean) ** 2 for x in data]
        return sum(squared_diffs) / len(data)
    
    def calculate_std_dev(data: List[float]) -> float:
        """Standard deviation: square root of variance."""
        return calculate_variance(data) ** 0.5
    
    # Calculate for both datasets
    print("Consistent Student:")
    print(f"  Scores: {consistent_scores}")
    print(f"  Mean: {calculate_mean(consistent_scores):.2f}")
    print(f"  Variance: {calculate_variance(consistent_scores):.2f}")
    print(f"  Std Dev: {calculate_std_dev(consistent_scores):.2f}")
    
    print("\nVariable Student:")
    print(f"  Scores: {variable_scores}")
    print(f"  Mean: {calculate_mean(variable_scores):.2f}")
    print(f"  Variance: {calculate_variance(variable_scores):.2f}")
    print(f"  Std Dev: {calculate_std_dev(variable_scores):.2f}")
    
    print("\n💡 Same mean, different variance = different behavior!")
    print("   In ML: High variance features might need normalization")
    
    # Why Standard Deviation? The "Same Units" Problem
    print("\n" + "=" * 70)
    print("🤔 Why Standard Deviation? Why Not Just Use Variance?")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Units Problem")
    print()
    print("   Scenario: Measuring daily commute time")
    print("   Data: [28, 30, 32, 29, 31] minutes")
    print()
    
    commute_times = [28, 30, 32, 29, 31]
    mean_commute = calculate_mean(commute_times)
    var_commute = calculate_variance(commute_times)
    std_commute = calculate_std_dev(commute_times)
    
    print(f"   Mean: {mean_commute:.1f} minutes")
    print(f"   Variance: {var_commute:.1f} minutes²  ← Wait, SQUARED minutes?!")
    print(f"   Std Dev: {std_commute:.1f} minutes   ← Back to regular minutes!")
    print()
    print("   ❓ What does '2.0 squared minutes' mean in real life?")
    print("      Can't visualize it! Can't compare it to your commute time!")
    print()
    print("   ✅ But '1.4 minutes' makes sense:")
    print("      'Your commute varies by about ±1.4 minutes from average'")
    print("      This is something you can FEEL and UNDERSTAND!")
    print()
    
    print("📊 Real Comparison:")
    print()
    print("   Your Commute: 30 min ± 1.4 min")
    print("   Friend's Commute: 30 min ± 15 min")
    print()
    print("   Now you can compare apples-to-apples:")
    print("   • You: Arrive within 28.6-31.4 min (predictable!)")
    print("   • Friend: Arrive within 15-45 min (chaos!)")
    print()
    
    print("💡 THREE REASONS We Need Standard Deviation:")
    print()
    print("   1️⃣  INTERPRETABLE UNITS:")
    print("      Variance: minutes² (makes no sense)")
    print("      Std Dev: minutes (you can understand it!)")
    print()
    print("   2️⃣  COMPARABLE TO DATA:")
    print("      Data: 28-32 minutes range")
    print("      Variance: 2.0 minutes² (can't compare!)")
    print("      Std Dev: 1.4 minutes (ah, small spread!)")
    print()
    print("   3️⃣  USEFUL RULES OF THUMB:")
    print("      Normal distribution: ~68% of data within 1 std dev")
    print("      Example: Mean=30, Std=1.4")
    print("      → 68% of days: 28.6-31.4 min commute")
    print("      → 95% of days: 27.2-32.8 min commute")
    print("      You can plan your day around this!")
    print()
    
    # Real ML Example
    print("🤖 ML Application: Model Error Analysis")
    print()
    print("   Your model predicts house prices:")
    print("   Mean error: $5,000 (not too bad!)")
    print("   Variance: $25,000,000 (wait, what does this mean?!)")
    print("   Std Dev: $5,000 (ah, typically off by ±$5k)")
    print()
    print("   With Std Dev, you can tell stakeholders:")
    print("   'Model predicts within ±$5,000 on average'")
    print()
    print("   Confidence range: Predicted price ± 2*std")
    print("   Example: Model says $200,000")
    print(f"           95% confident true price: $190k-$210k")
    print()
    
    print("🎯 WHEN TO USE EACH:")
    print("   • Variance: Math/formulas (e.g., calculating statistics)")
    print("   • Std Dev: Understanding/communicating (e.g., reporting results)")
    print()
    print("   Think of variance as 'calculation tool'")
    print("   Think of std dev as 'interpretation tool'")
    print()
    
    # Real-World Application: Model Performance
    print("\n🤖 Real Example: Two ML Models with Same Accuracy")
    print("   Model A: 85%, 86%, 84%, 85%, 85% (consistent)")
    print("   Model B: 60%, 95%, 70%, 100%, 75% (unpredictable)")
    print("   Both average 85%, but which would YOU trust in production?")
    print("   → Model A! Low variance = predictable = reliable")
    print("\n   Real Impact:")
    print("   • Model A: Customer sees consistent experience")
    print("   • Model B: Sometimes amazing, sometimes terrible (angry customers!)")
    print("   • In prod: Pick low variance even if mean is slightly lower")
    
    # Real ML Example: Feature Scaling
    print("\n📊 ML Application: Feature Scaling")
    print("   Feature 1 (age): Mean=30, Std=10")
    print("   Feature 2 (income): Mean=50000, Std=20000")
    print("   Problem: Income dominates due to scale!")
    print("   Solution: Standardize both to mean=0, std=1")
    
    ages = [25, 30, 35, 28, 32]
    mean_age = calculate_mean(ages)
    std_age = calculate_std_dev(ages)
    
    # Z-score normalization
    normalized_ages = [(age - mean_age) / std_age for age in ages]
    print(f"\n   Original ages: {ages}")
    print(f"   Normalized: {[f'{x:.2f}' for x in normalized_ages]}")
    print("   Now: Mean≈0, Std≈1")


# ============================================================================
# 2. Distributions
# ============================================================================

def demo_distributions():
    """
    Distribution: How data is spread across possible values.
    
    INTUITION:
    Imagine standing at a busy coffee shop, tracking customer behavior:
    
    - Uniform: Coffee sizes ordered (S/M/L) - all equally popular
      → Like a fair dice, each option has equal chance
    
    - Normal: Customer ages - most are 25-35, fewer teenagers/retirees
      → Bell curve! Heights, test scores, errors all follow this
    
    - Binomial: Out of 10 customers, how many order pastries?
      → Counting successes in fixed trials (ad clicks, email opens)
    
    WHY IT MATTERS IN ML:
    Know your data's shape → Pick the right model & loss function!
    - Normal data? Use MSE loss
    - Rare events? Use binary cross-entropy
    - Initialize weights? Sample from normal distribution
    
    ML Use Cases:
    - Understanding data patterns
    - Choosing loss functions
    - Initializing neural network weights
    """
    print("\n" + "=" * 70)
    print("2. Probability Distributions")
    print("=" * 70)
    print()
    print("💭 INTUITION: Picture a busy coffee shop")
    print("   Uniform: All coffee sizes equally popular (fair choice)")
    print("   Normal: Most customers aged 25-35 (bell curve)")
    print("   Binomial: Out of 10 customers, how many buy pastries?")
    print()
    
    # Simulate dice rolls (uniform distribution)
    print("🎲 Uniform Distribution (Fair Dice):")
    rolls = [random.randint(1, 6) for _ in range(1000)]
    counts = Counter(rolls)
    
    for value in sorted(counts.keys()):
        bar = "█" * (counts[value] // 10)
        print(f"   {value}: {bar} ({counts[value]})")
    
    print("   Each outcome ≈ equally likely (uniform)")
    
    # Simulate heights (normal distribution approximation)
    print("\n📏 Normal Distribution (Heights in cm):")
    print("   Most people near average, fewer at extremes")
    
    def generate_normal_approx(mean: float, std: float, n: int) -> List[float]:
        """Approximate normal distribution using central limit theorem."""
        # Sum of uniform random variables → normal distribution
        samples = []
        for _ in range(n):
            # Sum 12 uniform [0,1] random numbers
            uniform_sum = sum(random.random() for _ in range(12))
            # Adjust to desired mean and std
            value = mean + (uniform_sum - 6) * std
            samples.append(value)
        return samples
    
    heights = generate_normal_approx(mean=170, std=10, n=1000)
    
    # Count in buckets
    buckets = {}
    for h in heights:
        bucket = int(h // 5) * 5  # Round to nearest 5
        buckets[bucket] = buckets.get(bucket, 0) + 1
    
    for bucket in sorted(buckets.keys()):
        if 150 <= bucket <= 190:  # Show relevant range
            bar = "█" * (buckets[bucket] // 10)
            print(f"   {bucket:3d}cm: {bar}")
    
    print("\n💡 Normal distribution: Bell curve, symmetric around mean")
    print("   In ML: Many features naturally follow this pattern")
    
    # Coin flips (binomial distribution)
    print("\n🪙 Binomial Distribution (10 Coin Flips):")
    print("   Count heads in 10 flips, repeat 1000 times")
    
    experiments = []
    for _ in range(1000):
        flips = [random.choice([0, 1]) for _ in range(10)]
        heads = sum(flips)
        experiments.append(heads)
    
    counts = Counter(experiments)
    for heads in sorted(counts.keys()):
        bar = "█" * (counts[heads] // 20)
        print(f"   {heads:2d} heads: {bar} ({counts[heads]})")
    
    print("\n💡 Binomial: Most likely outcome near p*n (here 5 heads)")
    
    # Real-World Applications
    print("\n🌍 Real-World Distribution Examples:")
    print("\n1. Uniform → Random user ID assignment")
    print("   Users get IDs 1-1000, each equally likely")
    print("   Use case: Load balancing across servers")
    
    print("\n2. Normal → User engagement time")
    print("   Most users: 5-7 min/day (average behavior)")
    print("   Few users: <1 min or >15 min (outliers)")
    print("   Use case: Detect power users vs churning users")
    
    print("\n3. Binomial → Email campaign success")
    print("   Send 1000 emails, 20% click rate")
    print("   Expect: ~200 clicks ± some variance")
    print("   Use case: A/B test email subject lines")
    
    print("\n💡 WHY THIS MATTERS:")
    print("   • Know your distribution → Pick right loss function")
    print("   • Normal data? MSE loss works great")
    print("   • Imbalanced classes? Use weighted cross-entropy")
    print("   • Rare events? Focal loss or class weights")


# ============================================================================
# 3. Bayes' Theorem - The Heart of ML
# ============================================================================

def demo_bayes_theorem():
    """
    Bayes' Theorem: Update beliefs based on new evidence.
    
    Formula: P(A|B) = P(B|A) * P(A) / P(B)
    
    INTUITION - The Medical Test Scenario:
    You test positive for a rare disease (1% of people have it).
    The test is 95% accurate. Should you panic?
    
    NO! Here's why:
    - 1% actually have disease (rare)
    - 95% true positive rate (if you have it, test catches it)
    - 5% false positive rate (if healthy, test wrong 5% of time)
    
    Out of 10,000 people:
    - 100 have disease → 95 test positive (true positives)
    - 9,900 are healthy → 495 test positive (false positives!)
    
    So if you test positive:
    - 95 true positives / (95 + 495) total positives = 16% chance
    - You're probably fine! Get retested.
    
    THE KEY INSIGHT:
    Base rate matters! Rare disease + imperfect test = lots of false alarms.
    This is exactly how spam filters work - updating probability with each word.
    
    ML Use Cases:
    - Spam filters (update probability per word)
    - Medical diagnosis (test results + symptoms)
    - Recommendation systems (user behavior updates preferences)
    - Naive Bayes classifier (combine multiple features)
    """
    print("\n" + "=" * 70)
    print("3. Bayes' Theorem")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Medical Test Paradox")
    print("   You test positive for a rare disease (1% have it).")
    print("   Test is 95% accurate. Time to panic?")
    print()
    print("   NO! Out of 10,000 people:")
    print("   • 100 have disease → 95 test positive ✓")
    print("   • 9,900 healthy → 495 false positives! ✗")
    print("   → Only 95/(95+495) = 16% chance you actually have it!")
    print()
    print("   KEY: Base rate + test accuracy = actual probability")
    print("   Same math powers spam filters & recommendation engines!")
    print()
    
    print("🎯 Real Problem: Email Spam Detection")
    print()
    
    # Scenario: Word "FREE" appears in email
    # Question: Is it spam?
    
    # Given data:
    p_spam = 0.3                # 30% of emails are spam
    p_not_spam = 0.7            # 70% are legitimate
    p_free_given_spam = 0.8     # 80% of spam contains "FREE"
    p_free_given_not_spam = 0.1 # 10% of legit emails contain "FREE"
    
    # Calculate P(FREE) using law of total probability
    p_free = (p_free_given_spam * p_spam + 
              p_free_given_not_spam * p_not_spam)
    
    # Apply Bayes' Theorem
    # P(Spam | FREE) = P(FREE | Spam) * P(Spam) / P(FREE)
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print(f"Given information:")
    print(f"  P(Spam) = {p_spam:.1%}           (30% of all emails)")
    print(f"  P(FREE|Spam) = {p_free_given_spam:.1%}    (80% of spam has 'FREE')")
    print(f"  P(FREE|Not Spam) = {p_free_given_not_spam:.1%} (10% of legit has 'FREE')")
    print()
    print(f"Calculated:")
    print(f"  P(FREE) = {p_free:.1%}")
    print(f"  P(Spam|FREE) = {p_spam_given_free:.1%}")
    print()
    print(f"✅ If email contains 'FREE', it's {p_spam_given_free:.1%} likely spam!")
    
    # Multiple words (Naive Bayes)
    print("\n📧 Multiple Words (Naive Bayes Approach):")
    
    email_words = ["FREE", "WINNER", "CLICK"]
    
    # Probabilities for each word
    word_probs_spam = {"FREE": 0.8, "WINNER": 0.7, "CLICK": 0.6}
    word_probs_legit = {"FREE": 0.1, "WINNER": 0.05, "CLICK": 0.2}
    
    # Naive assumption: words are independent
    # P(Spam | words) ∝ P(words | Spam) * P(Spam)
    prob_spam = p_spam
    prob_legit = p_not_spam
    
    for word in email_words:
        prob_spam *= word_probs_spam.get(word, 0.5)
        prob_legit *= word_probs_legit.get(word, 0.5)
    
    # Normalize
    total = prob_spam + prob_legit
    prob_spam_final = prob_spam / total
    
    print(f"  Email words: {email_words}")
    print(f"  P(Spam | words) = {prob_spam_final:.1%}")
    print()
    print("\n💡 Bayes updates probability as we see more evidence!")
    print("   This is how spam filters actually work!")
    
    # More Real-World Examples
    print("\n" + "=" * 70)
    print("More Real-World Bayes Examples")
    print("=" * 70)
    
    print("\n🏥 Medical Diagnosis (Why doctors order multiple tests):")
    print("   Patient complains of chest pain.")
    print("   • Base rate: 5% of patients have heart disease")
    print("   • Test 1 positive: Updates to 30% probability")
    print("   • Test 2 positive: Updates to 70% probability")
    print("   • Symptom check positive: Updates to 90% probability")
    print("   Each piece of evidence updates our belief!")
    
    print("\n🎮 Fraud Detection (Why you get account alerts):")
    print("   Normal: You buy coffee in NYC every morning")
    print("   Suddenly: $5000 purchase in Tokyo at 3 AM")
    print("   • Base rate: 0.1% of transactions are fraud")
    print("   • Foreign country: 10x more likely fraud")
    print("   • Odd hour: 5x more likely fraud")
    print("   • Large amount: 20x more likely fraud")
    print("   Combined: ~90% chance fraud → Block card!")
    
    print("\n🎯 Recommendation Systems (Why Netflix 'gets' you):")
    print("   You watch sci-fi movies → Update: Likely sci-fi fan")
    print("   You skip rom-coms → Update: Probably not interested")
    print("   You binge-watch Stranger Things → Update: Love this genre!")
    print("   Every action updates your taste profile (Bayesian learning)")
    
    print("\n💡 THE PATTERN: Start with baseline → Update with evidence")
    print("   This is the foundation of machine learning!")


# ============================================================================
# 4. Correlation vs Causation
# ============================================================================

def demo_correlation():
    """
    Correlation: How two variables change together (-1 to +1)
    
    INTUITION - Your Social Media Habits:
    
    Positive correlation (+1):
    - More time on Instagram → More ads you see
    - They move together in same direction
    
    Negative correlation (-1):
    - More time exercising → Less time on couch
    - They move in opposite directions
    
    No correlation (0):
    - Your shoe size ↔ Your coding skills
    - Completely unrelated
    
    THE BIG TRAP - Correlation ≠ Causation:
    
    Famous example: Nicolas Cage movies ↔ Pool drownings
    Strong correlation! But Nicolas doesn't cause drownings.
    Hidden factor: Summer (more movies release + more swimming)
    
    In ML: Use correlation to find useful features, but:
    - High correlation? → Good for predictions
    - Causation? → Requires experiments (A/B tests)
    
    ML Use Cases:
    - Feature selection (drop correlated features)
    - Understanding relationships (which features matter)
    - Detecting multicollinearity (features too similar)
    """
    print("\n" + "=" * 70)
    print("4. Correlation vs Causation")
    print("=" * 70)
    print()
    print("💭 INTUITION: Your Daily Habits")
    print("   Positive (+1): More coffee → More productivity")
    print("                  (move together, same direction)")
    print()
    print("   Negative (-1): More Netflix → Less sleep")
    print("                  (move together, opposite directions)")
    print()
    print("   No correlation (0): Shoe size ↔ Coding skills")
    print("                       (completely unrelated)")
    print()
    print("   ⚠️  THE TRAP: Ice cream sales ↔ Shark attacks")
    print("      Strong correlation but ice cream doesn't cause sharks!")
    print("      Hidden factor: Summer (hot weather)")
    print()
    
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Covariance
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        # Standard deviations
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        
        return cov / (n * std_x * std_y)
    
    # Example 1: House size vs price (positive correlation)
    house_size = [1000, 1200, 1500, 1800, 2000, 2200, 2500]
    house_price = [200, 230, 280, 340, 380, 420, 480]
    
    corr_house = calculate_correlation(house_size, house_price)
    print(f"🏠 House Size vs Price:")
    print(f"   Correlation: {corr_house:.3f}")
    print(f"   Interpretation: Strong positive (bigger → more expensive)")
    
    # Example 2: Exercise vs weight (negative correlation)
    exercise_hours = [0, 1, 2, 3, 4, 5, 6]
    weight_kg = [90, 88, 85, 82, 79, 77, 75]
    
    corr_exercise = calculate_correlation(exercise_hours, weight_kg)
    print(f"\n🏃 Exercise vs Weight:")
    print(f"   Correlation: {corr_exercise:.3f}")
    print(f"   Interpretation: Strong negative (more exercise → less weight)")
    
    # Example 3: Random (no correlation)
    random_x = [random.random() for _ in range(100)]
    random_y = [random.random() for _ in range(100)]
    
    corr_random = calculate_correlation(random_x, random_y)
    print(f"\n🎲 Random vs Random:")
    print(f"   Correlation: {corr_random:.3f}")
    print(f"   Interpretation: No correlation (close to 0)")
    
    # Correlation does NOT mean causation!
    print("\n⚠️  IMPORTANT: Correlation ≠ Causation")
    print()
    print("   Famous example:")
    print("   Ice cream sales ↔ Drowning deaths")
    print("   Strong correlation, but ice cream doesn't cause drowning!")
    print("   Hidden variable: Summer (hot weather)")
    print()
    print("\n   In ML: Use correlation for feature selection,")
    print("   but don't assume causality!")
    
    # More correlation traps
    print("\n" + "=" * 70)
    print("More Correlation Traps (Learn from these!)")
    print("=" * 70)
    
    print("\n🎓 Study Time vs Grades:")
    print("   Strong positive correlation ✓")
    print("   Causation? Probably YES (studying helps!)")
    print("   But also: Smart students study smarter, not just more")
    
    print("\n🏊 Swimming Pool Size vs Drowning Risk:")
    print("   Positive correlation ✓")
    print("   Causation? NO! Hidden factor: Wealth")
    print("   Rich families have pools AND water activities")
    
    print("\n📱 Screen Time vs Depression:")
    print("   Strong correlation ✓")
    print("   Causation? UNCLEAR!")
    print("   • Does screen time cause depression?")
    print("   • Or do depressed people use phones more?")
    print("   • Or does isolation cause both?")
    print("   → Need experiments (A/B tests) to prove causation!")
    
    print("\n💼 ML Feature Selection (Practical Use):")
    print("   Scenario: Predicting house prices")
    print("   • Square footage ↔ Price: r=0.85 → Keep it!")
    print("   • Square footage ↔ # Bedrooms: r=0.92 → Pick one (redundant)")
    print("   • Paint color ↔ Price: r=0.02 → Drop it (noise)")
    print("   Rule: Keep features with high correlation to target,")
    print("         Drop features highly correlated to each other")
    
    print("\n🔍 Detecting Multicollinearity:")
    income_data = [50000, 60000, 70000, 80000, 90000]
    spending_data = [45000, 54000, 63000, 72000, 81000]
    corr_income_spending = calculate_correlation(income_data, spending_data)
    print(f"   Income ↔ Spending: r={corr_income_spending:.3f}")
    print("   Problem: These features are almost identical!")
    print("   Solution: Keep one, drop the other (or use PCA)")
    print("   Why? Model can't tell which feature matters")


# ============================================================================
# 5. Sampling and Confidence Intervals
# ============================================================================

def demo_sampling():
    """
    Sampling: Estimating population properties from a sample
    Confidence Interval: Range where true value likely falls
    
    INTUITION - Restaurant Reviews:
    
    You want to know if a restaurant is good, but it has 10,000 reviews.
    You don't read all 10,000! You sample:
    
    - Sample 10 reviews: Might get lucky/unlucky batch → unreliable
    - Sample 100 reviews: Better picture, but still some noise
    - Sample 500 reviews: Pretty confident in the average rating
    
    Confidence Interval = "I'm 95% sure the true rating is 4.2-4.6 stars"
    
    THE PRINCIPLE:
    Larger sample → Tighter confidence interval → More certain
    
    In ML, this means:
    - Small training data? Model might be lucky/unlucky (high uncertainty)
    - Large training data? Model performance is reliable
    - Test on 10 examples? Could be misleading
    - Test on 1000 examples? Trust the accuracy score
    
    Real Example - A/B Testing:
    "New button increased clicks by 5%"
    With 100 users: Maybe just luck
    With 10,000 users: Statistically significant, ship it!
    
    ML Use Cases:
    - Train/test split (sample to estimate real performance)
    - Cross-validation (multiple samples for robust estimate)
    - A/B testing (is the difference real or random?)
    - Model performance estimation (how confident in accuracy?)
    """
    print("\n" + "=" * 70)
    print("5. Sampling and Confidence Intervals")
    print("=" * 70)
    print()
    print("💭 INTUITION: Reading Restaurant Reviews")
    print("   Restaurant has 10,000 reviews. You can't read them all!")
    print()
    print("   Sample 10 reviews: Might get lucky/unlucky batch")
    print("   Sample 100 reviews: Better picture, still some noise")
    print("   Sample 500 reviews: Very confident in true rating")
    print()
    print("   Confidence Interval: 'I'm 95% sure true rating is 4.2-4.6 ⭐'")
    print()
    print("   In ML: Larger sample → More certain about model performance")
    print()
    
    print("🎯 Problem: Estimate average user age (can't survey everyone)")
    print()
    
    # True population (we pretend we don't know this)
    population = [random.randint(18, 65) for _ in range(10000)]
    true_mean = sum(population) / len(population)
    
    print(f"True population mean: {true_mean:.1f} years (unknown in practice)")
    print()
    
    # Take samples and estimate
    sample_sizes = [10, 50, 100, 500]
    
    for size in sample_sizes:
        # Take random sample
        sample = random.sample(population, size)
        sample_mean = sum(sample) / len(sample)
        
        # Calculate standard error (std dev of sample mean)
        sample_std = (sum((x - sample_mean) ** 2 for x in sample) / size) ** 0.5
        std_error = sample_std / (size ** 0.5)
        
        # 95% confidence interval (approximately mean ± 2*SE)
        margin = 1.96 * std_error
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin
        
        print(f"Sample size: {size:4d}")
        print(f"  Sample mean: {sample_mean:.1f}")
        print(f"  95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
        print(f"  Contains true mean? {'✅' if ci_lower <= true_mean <= ci_upper else '❌'}")
        print()
    
    print("💡 Larger samples → More accurate estimates!")
    print("   In ML: More training data → Better model performance")
    
    # Train/Test Split (sampling in ML)
    print("\n📊 ML Application: Train/Test Split")
    
    dataset_size = 1000
    train_ratio = 0.8
    
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    
    split_point = int(dataset_size * train_ratio)
    train_indices = all_indices[:split_point]
    test_indices = all_indices[split_point:]
    
    print(f"  Dataset: {dataset_size} samples")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/dataset_size:.1%})")
    print(f"  Test: {len(test_indices)} samples ({len(test_indices)/dataset_size:.1%})")
    print()
    print("  Why? Train on some data, test on unseen data")
    print("  Ensures model generalizes, not just memorizes!")
    
    # A/B Testing Example
    print("\n" + "=" * 70)
    print("A/B Testing: Is the Difference Real or Luck?")
    print("=" * 70)
    
    print("\n🎯 Scenario: Testing New Website Button")
    print("   Current (A): Blue button, 10% click rate")
    print("   New (B): Red button, 12% click rate")
    print("   Question: Is 2% improvement real or random luck?")
    print()
    
    # Simulate A/B test
    def run_ab_test(n_users: int, true_rate_a: float, true_rate_b: float) -> None:
        """Simulate A/B test and check if difference is significant."""
        # Simulate clicks
        clicks_a = sum(1 for _ in range(n_users) if random.random() < true_rate_a)
        clicks_b = sum(1 for _ in range(n_users) if random.random() < true_rate_b)
        
        rate_a = clicks_a / n_users
        rate_b = clicks_b / n_users
        
        # Standard error for difference in proportions
        se_a = (rate_a * (1 - rate_a) / n_users) ** 0.5
        se_b = (rate_b * (1 - rate_b) / n_users) ** 0.5
        se_diff = (se_a ** 2 + se_b ** 2) ** 0.5
        
        # Confidence interval for difference
        diff = rate_b - rate_a
        margin = 1.96 * se_diff
        ci_lower = diff - margin
        ci_upper = diff + margin
        
        # Significant if CI doesn't include 0
        significant = ci_lower > 0
        
        print(f"  Users per variant: {n_users:,}")
        print(f"  A: {clicks_a}/{n_users} = {rate_a:.1%} click rate")
        print(f"  B: {clicks_b}/{n_users} = {rate_b:.1%} click rate")
        print(f"  Difference: {diff:.1%}")
        print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
        print(f"  Statistically significant? {'✅ YES - Ship it!' if significant else '❌ NO - Need more data'}")
        print()
    
    print("Test 1: Small sample (100 users each)")
    run_ab_test(100, 0.10, 0.12)
    
    print("Test 2: Medium sample (1,000 users each)")
    run_ab_test(1000, 0.10, 0.12)
    
    print("Test 3: Large sample (10,000 users each)")
    run_ab_test(10000, 0.10, 0.12)
    
    print("💡 Key Insights:")
    print("   • Small sample: Result unclear (could be luck)")
    print("   • Large sample: Clear winner (not luck!)")
    print("   • In production: Wait for statistical significance")
    print("   • Rule of thumb: Need 1000+ users per variant")
    print()
    print("⚠️  Common Mistake: 'Button B won after 50 users!'")
    print("   → Too early! Could just be lucky batch of users.")
    print("   → Wait for confidence interval to not include 0.")


# ============================================================================
# 6. Real-World: Feature Normalization
# ============================================================================

def demo_feature_normalization():
    """
    Real-world ML: Normalize features for better model training.
    
    INTUITION - The Shouting Problem:
    
    You're in a meeting with:
    - Quiet colleague (speaking at volume 3/10)
    - Loud colleague (SHOUTING at volume 10/10)
    
    Who gets heard? The loud one, even if quiet one has better ideas!
    
    Same problem in ML:
    - Feature 1 (age): 25-65 (small numbers)
    - Feature 2 (salary): $30,000-$200,000 (BIG numbers)
    
    Without normalization:
    Model: "Salary changed by $1000? HUGE UPDATE!"
    Model: "Age changed by 1 year? Meh, barely notice."
    
    But maybe age matters MORE for your prediction!
    
    With normalization:
    Both features: -2 to +2 range
    Model: "Now I can hear both features equally!"
    
    Real Impact:
    - Training is 10-100x faster (gradient descent converges)
    - Model actually learns from ALL features, not just loud ones
    - Prediction accuracy improves
    
    Why?
    - Features have different scales (age: 0-100, income: 0-1000000)
    - Large values dominate gradient descent
    - Models converge faster with normalized features
    """
    print("\n" + "=" * 70)
    print("6. Real-World: Feature Normalization")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Shouting Problem in a Meeting")
    print("   Quiet person (volume 3/10): Has great ideas")
    print("   Loud person (VOLUME 10/10): Mediocre ideas")
    print("   Who gets heard? The loud one!")
    print()
    print("   Same in ML:")
    print("   Age: 25-65 (quiet numbers)")
    print("   Salary: $30k-$200k (SHOUTING NUMBERS!)")
    print()
    print("   Without normalization: Model only 'hears' salary")
    print("   With normalization: Both features get equal voice")
    print("   Result: Better predictions + 10-100x faster training!")
    print()
    
    # Sample data: house features
    houses = [
        {"size": 1000, "age": 5, "price": 200000},
        {"size": 1500, "age": 10, "price": 300000},
        {"size": 2000, "age": 2, "price": 400000},
        {"size": 2500, "age": 15, "price": 500000},
        {"size": 3000, "age": 8, "price": 600000},
    ]
    
    print("Original features:")
    print("  Size (sqft): 1000-3000")
    print("  Age (years): 2-15")
    print("  Problem: Size dominates because of scale!")
    print()
    
    # Extract features
    sizes = [h["size"] for h in houses]
    ages = [h["age"] for h in houses]
    
    # Method 1: Standardization (z-score)
    def standardize(values: List[float]) -> List[float]:
        """Transform to mean=0, std=1."""
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return [(x - mean) / std for x in values]
    
    # Method 2: Min-Max Scaling
    def min_max_scale(values: List[float]) -> List[float]:
        """Transform to range [0, 1]."""
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    # Apply standardization
    sizes_standardized = standardize(sizes)
    ages_standardized = standardize(ages)
    
    print("After Standardization (z-score):")
    print("  Size:", [f"{x:.2f}" for x in sizes_standardized])
    print("  Age:", [f"{x:.2f}" for x in ages_standardized])
    print("  Now both centered at 0, similar scale!")
    print()
    
    # Apply min-max scaling
    sizes_minmax = min_max_scale(sizes)
    ages_minmax = min_max_scale(ages)
    
    print("After Min-Max Scaling:")
    print("  Size:", [f"{x:.2f}" for x in sizes_minmax])
    print("  Age:", [f"{x:.2f}" for x in ages_minmax])
    print("  Now both in [0, 1] range!")
    print()
    
    print("💡 When to use:")
    print("   Standardization: Most cases, especially with outliers")
    print("   Min-Max: When you need specific range [0,1]")
    
    # Real Impact Demonstration
    print("\n" + "=" * 70)
    print("Real Impact: Training Speed")
    print("=" * 70)
    
    print("\n🐌 WITHOUT Normalization:")
    print("   Gradient Descent:")
    print("   • Age updates: Tiny steps (small numbers)")
    print("   • Salary updates: HUGE jumps (big numbers)")
    print("   • Result: Zigzag path, slow convergence")
    print("   • Iterations to converge: 10,000+")
    print("   • Training time: 10 minutes")
    
    print("\n🚀 WITH Normalization:")
    print("   Gradient Descent:")
    print("   • Age updates: Balanced")
    print("   • Salary updates: Balanced")
    print("   • Result: Smooth path, fast convergence")
    print("   • Iterations to converge: 100-500")
    print("   • Training time: 30 seconds")
    
    print("\n💡 20x speedup just from normalization!")
    
    print("\n🎯 When Each Method Shines:")
    print("\n1. Standardization (Z-score):")
    print("   Use when:")
    print("   • Features have outliers (robust to extremes)")
    print("   • Using distance-based algorithms (KNN, SVM)")
    print("   • Neural networks (most common choice)")
    print("   Example: Age, income, test scores")
    
    print("\n2. Min-Max Scaling:")
    print("   Use when:")
    print("   • Need specific range [0,1] or [0,255]")
    print("   • Image pixel values (already 0-255)")
    print("   • No outliers in data")
    print("   Example: Pixel values, percentages, bounded data")
    
    print("\n3. Robust Scaling (mentioned for completeness):")
    print("   Use when:")
    print("   • Extreme outliers you want to keep")
    print("   • Uses median & IQR (less sensitive to outliers)")
    print("   Example: Financial data with rare huge transactions")


# ============================================================================
# 7. Real-World: Outlier Detection
# ============================================================================

def demo_outlier_detection():
    """
    Real-world ML: Detect unusual data points.
    
    INTUITION - Your Morning Alarm:
    
    You usually wake up at 7:00 AM (±10 minutes):
    - Monday: 7:05 AM ✓ Normal
    - Tuesday: 6:55 AM ✓ Normal  
    - Wednesday: 7:10 AM ✓ Normal
    - Thursday: 2:30 AM ⚠️ OUTLIER! (Cat stepped on phone)
    - Friday: 7:02 AM ✓ Normal
    
    Outliers are data points that don't fit the pattern.
    
    In ML/Production Systems:
    
    API Latency:
    - Normal: 45-55ms (fast, healthy)
    - Outlier: 3000ms (something's broken!)
    
    User Purchases:
    - Normal: $20-$200 orders
    - Outlier: $50,000 order (fraud? or genuine VIP?)
    
    Sensor Data:
    - Normal: Temperature 20-25°C
    - Outlier: -999°C (sensor malfunction!)
    
    Why Detect Outliers?
    1. Data Quality: Remove errors before training
    2. Fraud Detection: Flag suspicious transactions
    3. System Health: Alert on performance issues
    4. Model Protection: Outliers can skew your model
    
    In ML Training:
    Bad outlier: Sensor error reading -999°C → Remove it
    Good outlier: Rare disease case → Keep it (valuable edge case!)
    
    Why important?
    - Bad data (errors, sensor failures)
    - Fraud detection
    - Anomaly detection
    - Data quality
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Outlier Detection")
    print("=" * 70)
    print()
    print("💭 INTUITION: Your Morning Alarm")
    print("   You wake up at ~7:00 AM every day:")
    print("   Mon: 7:05 AM ✓  Normal")
    print("   Tue: 6:55 AM ✓  Normal")
    print("   Wed: 2:30 AM ⚠️  OUTLIER! (Cat stepped on phone)")
    print("   Thu: 7:10 AM ✓  Normal")
    print()
    print("   In Production:")
    print("   API: 45-55ms normal, 3000ms = SOMETHING BROKE!")
    print("   Purchase: $20-200 normal, $50,000 = Fraud or VIP?")
    print("   Sensor: 20-25°C normal, -999°C = Malfunction!")
    print()
    print("   Action: Remove errors, investigate anomalies")
    print()
    
    print("📊 Scenario: API Response Times (milliseconds)")
    print()
    
    # Normal response times with some outliers
    response_times = [
        45, 50, 48, 52, 49, 51, 47, 53, 50, 48,  # Normal
        51, 49, 52, 50, 48, 51, 49, 50, 47, 52,  # Normal
        250, 300,  # Outliers (something wrong!)
        48, 50, 49, 51, 50, 48, 52, 49  # Normal
    ]
    
    # Calculate statistics
    mean = sum(response_times) / len(response_times)
    variance = sum((x - mean) ** 2 for x in response_times) / len(response_times)
    std = variance ** 0.5
    
    print(f"Mean: {mean:.1f}ms")
    print(f"Std Dev: {std:.1f}ms")
    print()
    
    # Method: Z-score (how many std devs from mean)
    # If |z| > 3, likely outlier
    threshold = 3
    
    outliers = []
    normal = []
    
    for time in response_times:
        z_score = (time - mean) / std
        if abs(z_score) > threshold:
            outliers.append(time)
        else:
            normal.append(time)
    
    print(f"Normal responses: {len(normal)} (within {threshold}σ)")
    print(f"Outliers: {len(outliers)}")
    print(f"  Values: {outliers}ms")
    print()
    
    # Alternative: IQR method
    sorted_times = sorted(response_times)
    n = len(sorted_times)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    
    q1 = sorted_times[q1_idx]
    q3 = sorted_times[q3_idx]
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    iqr_outliers = [t for t in response_times if t < lower_bound or t > upper_bound]
    
    print("Alternative: IQR Method")
    print(f"  Q1: {q1}ms, Q3: {q3}ms")
    print(f"  IQR: {iqr}ms")
    print(f"  Valid range: [{lower_bound:.1f}, {upper_bound:.1f}]ms")
    print(f"  Outliers: {iqr_outliers}ms")
    print()
    
    print("💡 In production ML:")
    print("   - Remove outliers or cap them")
    print("   - Investigate root cause")
    print("   - Use robust models (less sensitive)")
    
    # Outlier Strategy Guide
    print("\n" + "=" * 70)
    print("Outlier Decision Framework")
    print("=" * 70)
    
    print("\n❓ Ask: Is this outlier REAL or ERROR?")
    print()
    
    print("🔍 Scenario 1: Sensor Reads -999°C")
    print("   Analysis: Physically impossible (absolute zero is -273°C)")
    print("   Decision: ❌ ERROR - Remove it")
    print("   Action: Filter out, log error, alert monitoring team")
    
    print("\n🔍 Scenario 2: User Spent $50,000 on App Purchase")
    print("   Analysis: Unusual but possible (whale customer or fraud)")
    print("   Decision: ⚠️  INVESTIGATE - Don't auto-remove")
    print("   Action: Keep in data, flag for fraud team, add as edge case")
    
    print("\n🔍 Scenario 3: Website Load Time 30 seconds")
    print("   Analysis: Rare but real (slow connection, server hiccup)")
    print("   Decision: ✅ REAL - Keep it")
    print("   Action: Include in training (model should handle this)")
    
    print("\n🔍 Scenario 4: Age = 150 years")
    print("   Analysis: Typo or data entry error (no human lives 150 years)")
    print("   Decision: ❌ ERROR - Remove or cap")
    print("   Action: Cap at reasonable max (e.g., 100) or filter out")
    
    print("\n" + "=" * 70)
    print("Outlier Handling Techniques")
    print("=" * 70)
    
    print("\n1️⃣  Remove Completely:")
    print("   When: Clear data errors, impossible values")
    print("   Example: Negative ages, temperature > 1000°C")
    print("   Code: df = df[df['age'] > 0]")
    
    print("\n2️⃣  Cap/Clip (Winsorization):")
    print("   When: Outliers are real but too extreme")
    print("   Example: Cap income at 99th percentile")
    demo_values = [10, 20, 25, 30, 35, 40, 45, 1000]
    p99 = sorted(demo_values)[int(len(demo_values) * 0.99)]
    capped = [min(x, p99) for x in demo_values]
    print(f"   Before: {demo_values}")
    print(f"   After:  {capped}")
    
    print("\n3️⃣  Transform (Log/Sqrt):")
    print("   When: Data is skewed (e.g., income, website traffic)")
    print("   Example: log(income) makes distribution more normal")
    import math
    skewed = [1000, 5000, 10000, 50000, 1000000]
    logged = [math.log10(x) for x in skewed]
    print(f"   Original: {skewed}")
    print(f"   Log10:    {[f'{x:.2f}' for x in logged]}")
    print("   Now: Less spread, outliers less extreme")
    
    print("\n4️⃣  Keep & Use Robust Models:")
    print("   When: Outliers are valuable (fraud, rare diseases)")
    print("   Models: Random Forest, Gradient Boosting (less sensitive)")
    print("   Avoid: Linear Regression (very sensitive to outliers)")
    
    print("\n5️⃣  Separate Model for Outliers:")
    print("   When: Outliers are a different category")
    print("   Example: Separate fraud detection model")
    print("   Approach: Train 2 models - normal behavior + anomaly detection")
    
    print("\n💡 GOLDEN RULE:")
    print("   NEVER blindly remove outliers!")
    print("   Always investigate → Understand → Then decide")
    print("   Sometimes the outlier is your most valuable data point!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n📊 Probability & Statistics for ML/AI Engineering\n")
    print("Focus: Build intuition, not memorize formulas!")
    print()
    
    demo_mean_variance()
    demo_distributions()
    demo_bayes_theorem()
    demo_correlation()
    demo_sampling()
    demo_feature_normalization()
    demo_outlier_detection()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Mean: Center of data
2. Variance/Std: How spread out data is
3. Normal Distribution: Bell curve, most common
4. Bayes' Theorem: Update beliefs with evidence
5. Correlation: Relationship strength, NOT causation
6. Sampling: Estimate from subset
7. Confidence Intervals: Range for true value
8. Normalization: Scale features for ML
9. Outlier Detection: Find unusual data points

ML Applications:
- Feature scaling: Standardize before training
- Feature selection: Use correlation
- Spam detection: Naive Bayes
- Data quality: Outlier detection
- Model evaluation: Sampling, confidence intervals
- A/B testing: Statistical significance

Next: 02_linear_algebra.py - Vectors, matrices, dot products!
""")


if __name__ == "__main__":
    main()
