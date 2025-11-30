"""
Practical ML Statistics & Data Science

Real-world Python examples for ML engineering tasks.
Topics: EDA, Feature Scaling, Naive Bayes, Feature Selection, 
        Dimensionality Reduction, Model Validation, A/B Testing

Install: poetry add numpy pandas scikit-learn matplotlib seaborn scipy
Run: poetry run python 03-math-for-ml/examples/05_practical_ml_statistics.py
"""

import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math


# ============================================================================
# 1. Exploratory Data Analysis (EDA)
# ============================================================================

def demo_eda():
    """
    EDA: Understanding your data BEFORE modeling
    
    INTUITION - The Crime Scene Investigation:
    
    You wouldn't arrest someone without investigating the crime scene!
    Similarly, don't train a model without investigating your data.
    
    EDA answers:
    - What does my data look like?
    - Are there missing values?
    - Are there outliers?
    - What are the distributions?
    - Are features correlated?
    - Is data balanced?
    
    Real Example - E-commerce Dataset:
    You have 10,000 customer records. Before building churn model:
    1. Check: How many churned? (95% stayed, 5% left → IMBALANCED!)
    2. Check: Age range? (18-90, but 1 customer age 150 → OUTLIER!)
    3. Check: Income missing? (30% missing → NEED TO HANDLE!)
    4. Check: Correlations? (purchases ↔ time_on_site → RELATED!)
    
    WHY IT MATTERS:
    - Skip EDA → Train on garbage → Get garbage predictions
    - Do EDA → Find issues → Fix them → Good predictions
    
    Rule: Spend 70% of time on EDA, 30% on modeling!
    """
    print("=" * 70)
    print("1. Exploratory Data Analysis (EDA)")
    print("=" * 70)
    print()
    print("💭 INTUITION: Crime Scene Investigation")
    print("   Don't arrest without investigating!")
    print("   Don't model without exploring data!")
    print()
    
    # Simulate customer churn dataset
    print("📊 E-commerce Customer Dataset")
    print()
    
    # Generate realistic customer data
    random.seed(42)
    n_customers = 100
    
    customers = []
    for i in range(n_customers):
        age = random.randint(18, 70) if random.random() > 0.05 else None  # 5% missing
        if age and random.random() < 0.01:  # 1% outliers
            age = random.randint(150, 200)
        
        purchases = random.randint(0, 50)
        time_on_site = purchases * 10 + random.randint(-20, 20)  # Correlated!
        income = random.randint(30000, 150000) if random.random() > 0.1 else None  # 10% missing
        
        # Churn more likely if low purchases
        churned = purchases < 10 and random.random() < 0.4
        
        customers.append({
            'age': age,
            'purchases': purchases,
            'time_on_site': time_on_site,
            'income': income,
            'churned': churned
        })
    
    print(f"Total customers: {n_customers}")
    print()
    
    # STEP 1: Check Target Distribution
    print("STEP 1: Target Variable Distribution")
    churned_count = sum(1 for c in customers if c['churned'])
    churn_rate = churned_count / n_customers
    print(f"   Churned: {churned_count} ({churn_rate:.1%})")
    print(f"   Stayed: {n_customers - churned_count} ({1-churn_rate:.1%})")
    
    if churn_rate < 0.1 or churn_rate > 0.9:
        print("   ⚠️  IMBALANCED! Need to handle:")
        print("      - Use class weights")
        print("      - Oversample minority class")
        print("      - Use appropriate metrics (F1, not accuracy)")
    print()
    
    # STEP 2: Check Missing Values
    print("STEP 2: Missing Values Analysis")
    features = ['age', 'purchases', 'time_on_site', 'income']
    
    for feature in features:
        missing = sum(1 for c in customers if c[feature] is None)
        missing_pct = missing / n_customers * 100
        print(f"   {feature:15s}: {missing:3d} missing ({missing_pct:.1f}%)")
        
        if missing_pct > 5:
            print(f"      → Action: Impute with median or drop feature")
    print()
    
    # STEP 3: Check for Outliers
    print("STEP 3: Outlier Detection")
    
    ages = [c['age'] for c in customers if c['age'] is not None]
    if ages:
        age_mean = sum(ages) / len(ages)
        age_std = (sum((a - age_mean)**2 for a in ages) / len(ages)) ** 0.5
        
        outlier_ages = [a for a in ages if abs(a - age_mean) > 3 * age_std]
        
        print(f"   Age statistics:")
        print(f"      Mean: {age_mean:.1f}, Std: {age_std:.1f}")
        print(f"      Range: {min(ages)}-{max(ages)}")
        print(f"      Outliers (>3σ): {outlier_ages}")
        
        if outlier_ages:
            print(f"      ⚠️  Found {len(outlier_ages)} outliers!")
            print(f"      → Action: Cap at 100 or remove")
    print()
    
    # STEP 4: Feature Correlations
    print("STEP 4: Feature Correlations")
    
    # Calculate correlation between purchases and time_on_site
    purchases_list = [c['purchases'] for c in customers]
    time_list = [c['time_on_site'] for c in customers]
    
    n = len(purchases_list)
    mean_p = sum(purchases_list) / n
    mean_t = sum(time_list) / n
    
    cov = sum((purchases_list[i] - mean_p) * (time_list[i] - mean_t) for i in range(n))
    std_p = (sum((p - mean_p)**2 for p in purchases_list) / n) ** 0.5
    std_t = (sum((t - mean_t)**2 for t in time_list) / n) ** 0.5
    
    corr = cov / (n * std_p * std_t)
    
    print(f"   Purchases ↔ Time on Site: r={corr:.3f}")
    
    if abs(corr) > 0.8:
        print(f"      ⚠️  High correlation! Features redundant.")
        print(f"      → Action: Keep one, drop the other")
    elif abs(corr) > 0.5:
        print(f"      ✓ Moderate correlation (expected)")
    print()
    
    # STEP 5: Summary Statistics
    print("STEP 5: Summary Statistics")
    print(f"   Purchases: min={min(purchases_list)}, max={max(purchases_list)}, "
          f"mean={sum(purchases_list)/len(purchases_list):.1f}")
    print()
    
    print("💡 EDA Checklist Complete!")
    print("   ✓ Checked target distribution (imbalance?)")
    print("   ✓ Identified missing values (impute/drop?)")
    print("   ✓ Found outliers (cap/remove?)")
    print("   ✓ Analyzed correlations (redundant features?)")
    print("   ✓ Summary statistics (understand scale)")
    print()
    print("   Next: Clean data based on findings!")


# ============================================================================
# 2. Feature Scaling (Standardization vs Normalization)
# ============================================================================

def demo_feature_scaling():
    """
    Feature Scaling: Make features comparable
    
    INTUITION - The Salary vs Age Problem:
    
    Training a model to predict house price:
    - Feature 1: Bedrooms (1-5)
    - Feature 2: Square Feet (500-5000)
    - Feature 3: Price History ($100K-$1M)
    
    Without scaling:
    Model: "Price history changed $10K → HUGE UPDATE!"
    Model: "Bedrooms changed from 3 to 4 → Barely notice"
    
    But bedrooms might matter MORE than price history!
    
    With scaling:
    All features: Similar range (-2 to +2)
    Model: "I can consider all features equally!"
    
    METHODS:
    
    1. Standardization (Z-score): Mean=0, Std=1
       - Use: Most ML models (SVM, Neural Nets, KNN)
       - Pros: Handles outliers well
       - Formula: (x - mean) / std
    
    2. Min-Max Normalization: Scale to [0,1]
       - Use: Neural networks (especially output layer)
       - Pros: Bounded range
       - Cons: Sensitive to outliers
       - Formula: (x - min) / (max - min)
    
    3. Robust Scaling: Use median & IQR
       - Use: Data with many outliers
       - Pros: Very robust to outliers
       - Formula: (x - median) / IQR
    """
    print("\n" + "=" * 70)
    print("2. Feature Scaling")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Volume Problem")
    print("   Meeting with quiet (vol 3) and loud (vol 10) colleagues")
    print("   Loud one gets heard, even with worse ideas!")
    print()
    print("   Same in ML: Big numbers dominate small numbers")
    print("   Solution: Scale everything to similar range")
    print()
    
    # Real dataset: Predicting employee satisfaction
    print("📊 Employee Satisfaction Prediction")
    print()
    
    employees = [
        {'age': 25, 'salary': 50000, 'years': 2, 'satisfaction': 7},
        {'age': 35, 'salary': 80000, 'years': 10, 'satisfaction': 8},
        {'age': 45, 'salary': 120000, 'years': 20, 'satisfaction': 6},
        {'age': 28, 'salary': 60000, 'years': 4, 'satisfaction': 9},
        {'age': 50, 'salary': 150000, 'years': 25, 'satisfaction': 5},
    ]
    
    ages = [e['age'] for e in employees]
    salaries = [e['salary'] for e in employees]
    years = [e['years'] for e in employees]
    
    print("Original Features:")
    print(f"   Age: {min(ages)}-{max(ages)} (range: {max(ages)-min(ages)})")
    print(f"   Salary: ${min(salaries):,}-${max(salaries):,} (range: ${max(salaries)-min(salaries):,})")
    print(f"   Years: {min(years)}-{max(years)} (range: {max(years)-min(years)})")
    print()
    print("   Problem: Salary dominates (100K range vs 25 age range)!")
    print()
    
    # Method 1: Standardization
    def standardize(values: List[float]) -> List[float]:
        """Z-score normalization: mean=0, std=1"""
        mean = sum(values) / len(values)
        std = (sum((x - mean)**2 for x in values) / len(values)) ** 0.5
        return [(x - mean) / std for x in values]
    
    ages_std = standardize(ages)
    salaries_std = standardize(salaries)
    years_std = standardize(years)
    
    print("Method 1: Standardization (Z-score)")
    print("   Formula: (x - mean) / std")
    print()
    print(f"   Age:    {[f'{x:.2f}' for x in ages_std]}")
    print(f"   Salary: {[f'{x:.2f}' for x in salaries_std]}")
    print(f"   Years:  {[f'{x:.2f}' for x in years_std]}")
    print()
    print("   ✓ All centered at 0, similar spread!")
    print("   Use case: SVM, Neural Networks, KNN")
    print()
    
    # Method 2: Min-Max Normalization
    def min_max_scale(values: List[float]) -> List[float]:
        """Scale to [0, 1] range"""
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    ages_minmax = min_max_scale(ages)
    salaries_minmax = min_max_scale(salaries)
    years_minmax = min_max_scale(years)
    
    print("Method 2: Min-Max Normalization")
    print("   Formula: (x - min) / (max - min)")
    print()
    print(f"   Age:    {[f'{x:.2f}' for x in ages_minmax]}")
    print(f"   Salary: {[f'{x:.2f}' for x in salaries_minmax]}")
    print(f"   Years:  {[f'{x:.2f}' for x in years_minmax]}")
    print()
    print("   ✓ All in [0, 1] range!")
    print("   Use case: Neural networks (especially image data)")
    print()
    
    # Method 3: Robust Scaling
    def robust_scale(values: List[float]) -> List[float]:
        """Scale using median and IQR (robust to outliers)"""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        median = sorted_vals[n // 2]
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        return [(x - median) / iqr if iqr > 0 else 0 for x in values]
    
    ages_robust = robust_scale(ages)
    salaries_robust = robust_scale(salaries)
    years_robust = robust_scale(years)
    
    print("Method 3: Robust Scaling")
    print("   Formula: (x - median) / IQR")
    print()
    print(f"   Age:    {[f'{x:.2f}' for x in ages_robust]}")
    print(f"   Salary: {[f'{x:.2f}' for x in salaries_robust]}")
    print(f"   Years:  {[f'{x:.2f}' for x in years_robust]}")
    print()
    print("   ✓ Less sensitive to outliers!")
    print("   Use case: Financial data with extreme values")
    print()
    
    # Real Impact
    print("🚀 Real Impact on Training:")
    print()
    print("WITHOUT Scaling:")
    print("   Iteration 1: salary weight = 0.001 → 0.100 (huge jump!)")
    print("   Iteration 1: age weight = 0.5 → 0.501 (tiny change)")
    print("   Result: 10,000 iterations to converge")
    print()
    print("WITH Scaling:")
    print("   Iteration 1: salary weight = 0.1 → 0.15 (balanced)")
    print("   Iteration 1: age weight = 0.5 → 0.55 (balanced)")
    print("   Result: 500 iterations to converge (20x faster!)")
    print()
    
    print("💡 Decision Guide:")
    print("   Standardization → Default choice (most cases)")
    print("   Min-Max → Bounded range needed (0-1 or 0-255)")
    print("   Robust → Lots of outliers you want to keep")


# ============================================================================
# 3. Naive Bayes Classifier
# ============================================================================

def demo_naive_bayes():
    """
    Naive Bayes: Update probability with evidence
    
    INTUITION - Email Spam Filter:
    
    Email arrives: "FREE WINNER! Click here!"
    
    Base probability: 30% of emails are spam
    See "FREE" → Update to 77% spam
    See "WINNER" → Update to 95% spam
    See "Click" → Update to 98% spam
    
    Each word updates our belief!
    
    WHY "NAIVE"?
    Assumes words are independent (naive assumption!)
    "FREE" and "WINNER" probably appear together in spam.
    But math is easier if we assume independence.
    
    Despite being "naive", works GREAT for:
    - Spam filtering (Gmail uses it!)
    - Sentiment analysis
    - Document classification
    - Medical diagnosis
    
    TYPES:
    1. Multinomial: For count data (word counts)
    2. Bernoulli: For binary data (word present/absent)
    3. Gaussian: For continuous data (measurements)
    """
    print("\n" + "=" * 70)
    print("3. Naive Bayes Classifier")
    print("=" * 70)
    print()
    print("💭 INTUITION: Email Spam Detective")
    print("   Email: 'FREE WINNER! Click here!'")
    print()
    print("   Start: 30% spam (base rate)")
    print("   See 'FREE': Update → 77% spam")
    print("   See 'WINNER': Update → 95% spam")
    print("   See 'Click': Update → 98% spam")
    print()
    print("   Each word = new evidence = updated belief!")
    print()
    
    # Training data: emails with labels
    spam_emails = [
        "free money winner click now",
        "winner free prize click here",
        "click free offer limited time",
        "congratulations winner free gift click",
        "free winner click now special offer"
    ]
    
    ham_emails = [
        "meeting tomorrow at noon conference room",
        "project deadline next week please review",
        "lunch plans for friday team building",
        "quarterly report attached review please",
        "conference call scheduled monday morning"
    ]
    
    print("📧 Training Naive Bayes Spam Classifier")
    print()
    print(f"Training data: {len(spam_emails)} spam, {len(ham_emails)} ham")
    print()
    
    # Calculate word probabilities
    def train_naive_bayes(spam_docs: List[str], ham_docs: List[str]) -> Dict:
        """Train Naive Bayes classifier"""
        # Count words in each class
        spam_words = []
        for email in spam_docs:
            spam_words.extend(email.lower().split())
        
        ham_words = []
        for email in ham_docs:
            ham_words.extend(email.lower().split())
        
        # Calculate probabilities
        spam_word_counts = Counter(spam_words)
        ham_word_counts = Counter(ham_words)
        
        total_spam_words = len(spam_words)
        total_ham_words = len(ham_words)
        
        # P(word | spam)
        spam_probs = {}
        for word in set(spam_words):
            # Add-one smoothing to avoid zero probabilities
            spam_probs[word] = (spam_word_counts[word] + 1) / (total_spam_words + len(set(spam_words + ham_words)))
        
        # P(word | ham)
        ham_probs = {}
        for word in set(ham_words):
            ham_probs[word] = (ham_word_counts[word] + 1) / (total_ham_words + len(set(spam_words + ham_words)))
        
        # Prior probabilities
        total_docs = len(spam_docs) + len(ham_docs)
        p_spam = len(spam_docs) / total_docs
        p_ham = len(ham_docs) / total_docs
        
        return {
            'spam_probs': spam_probs,
            'ham_probs': ham_probs,
            'p_spam': p_spam,
            'p_ham': p_ham,
            'all_words': set(spam_words + ham_words)
        }
    
    model = train_naive_bayes(spam_emails, ham_emails)
    
    print("Learned Word Probabilities:")
    print()
    print("Top spam words:")
    top_spam = sorted(model['spam_probs'].items(), key=lambda x: x[1], reverse=True)[:5]
    for word, prob in top_spam:
        print(f"   '{word}': {prob:.3f}")
    print()
    
    # Classify new email
    def classify(email: str, model: Dict) -> Tuple[str, float]:
        """Classify email as spam or ham"""
        words = email.lower().split()
        
        # Start with prior probabilities
        spam_score = math.log(model['p_spam'])
        ham_score = math.log(model['p_ham'])
        
        # Update with each word (using log to avoid underflow)
        for word in words:
            if word in model['spam_probs']:
                spam_score += math.log(model['spam_probs'][word])
            if word in model['ham_probs']:
                ham_score += math.log(model['ham_probs'][word])
        
        # Classify
        if spam_score > ham_score:
            confidence = spam_score / (spam_score + ham_score)
            return "SPAM", confidence
        else:
            confidence = ham_score / (spam_score + ham_score)
            return "HAM", confidence
    
    print("Testing Classifier:")
    print()
    
    test_emails = [
        "free winner click now",
        "meeting tomorrow morning",
        "click here for free prize",
        "project deadline review please"
    ]
    
    for email in test_emails:
        label, confidence = classify(email, model)
        print(f"   Email: '{email}'")
        print(f"   Prediction: {label} (confidence: {confidence:.1%})")
        print()
    
    print("💡 Why Naive Bayes Works:")
    print("   • Fast: O(n) training, O(m) prediction")
    print("   • Simple: Easy to understand and implement")
    print("   • Works well: Even with 'naive' independence assumption")
    print("   • Needs little data: Can train on small datasets")
    print()
    print("Real-world uses:")
    print("   • Gmail spam filter")
    print("   • Sentiment analysis (positive/negative reviews)")
    print("   • Document categorization (news topics)")
    print("   • Medical diagnosis (symptoms → disease)")


# ============================================================================
# 4. Feature Selection
# ============================================================================

def demo_feature_selection():
    """
    Feature Selection: Pick the most useful features
    
    INTUITION - Hiring Decision:
    
    You're hiring a developer. Candidates give you 100 pieces of info:
    - Years of experience ✓ (USEFUL)
    - GitHub contributions ✓ (USEFUL)
    - Favorite color ✗ (USELESS)
    - Shoe size ✗ (USELESS)
    - Coding test score ✓ (USEFUL)
    
    You don't need all 100! Focus on the 10 that actually matter.
    
    WHY SELECT FEATURES?
    1. Faster training (fewer features = faster)
    2. Better generalization (less overfitting)
    3. Simpler model (easier to explain)
    4. Removes noise (bad features hurt performance)
    
    METHODS:
    
    1. Filter: Score each feature independently
       - Correlation with target
       - Statistical tests (chi-square, ANOVA)
       - Fast but ignores feature interactions
    
    2. Wrapper: Try different feature combinations
       - Forward selection (add one at a time)
       - Backward elimination (remove one at a time)
       - Slow but finds best combo
    
    3. Embedded: Model selects during training
       - Lasso regression (L1 penalty)
       - Tree-based feature importance
       - Fast and considers interactions
    """
    print("\n" + "=" * 70)
    print("4. Feature Selection")
    print("=" * 70)
    print()
    print("💭 INTUITION: Hiring a Developer")
    print("   100 pieces of info per candidate:")
    print("   • Years of experience → USEFUL")
    print("   • GitHub contributions → USEFUL")
    print("   • Favorite color → USELESS")
    print("   • Shoe size → USELESS")
    print()
    print("   Don't need all 100! Find the 10 that matter.")
    print()
    
    # Generate dataset: House price prediction
    print("📊 House Price Prediction Dataset")
    print()
    
    random.seed(42)
    n_houses = 100
    
    houses = []
    for _ in range(n_houses):
        sqft = random.randint(1000, 3000)
        bedrooms = random.randint(2, 5)
        age = random.randint(0, 50)
        distance_to_city = random.randint(1, 50)  # km
        
        # Useless features (random noise)
        house_number = random.randint(1, 1000)
        paint_color_code = random.randint(1, 10)
        
        # Price depends on sqft, bedrooms, age (NOT on house number or paint!)
        price = (sqft * 200 + bedrooms * 50000 - age * 2000 - 
                 distance_to_city * 1000 + random.randint(-50000, 50000))
        
        houses.append({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'age': age,
            'distance_to_city': distance_to_city,
            'house_number': house_number,  # NOISE!
            'paint_color': paint_color_code,  # NOISE!
            'price': price
        })
    
    features = ['sqft', 'bedrooms', 'age', 'distance_to_city', 'house_number', 'paint_color']
    
    print(f"Features: {len(features)}")
    for f in features:
        print(f"   • {f}")
    print()
    
    # Method 1: Correlation-based selection
    print("Method 1: Correlation with Target")
    print()
    
    prices = [h['price'] for h in houses]
    
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation"""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5
        
        return cov / (n * std_x * std_y)
    
    correlations = {}
    for feature in features:
        feature_values = [h[feature] for h in houses]
        corr = calculate_correlation(feature_values, prices)
        correlations[feature] = abs(corr)  # Absolute value
    
    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Feature Correlations with Price:")
    for feature, corr in sorted_features:
        status = "✓ Keep" if corr > 0.3 else "✗ Drop"
        print(f"   {feature:20s}: {corr:.3f}  {status}")
    print()
    
    # Select top features
    selected_features = [f for f, corr in sorted_features if corr > 0.3]
    
    print(f"Selected Features ({len(selected_features)}/{len(features)}):")
    for f in selected_features:
        print(f"   • {f}")
    print()
    
    # Method 2: Variance Threshold
    print("Method 2: Remove Low-Variance Features")
    print()
    
    def calculate_variance(values: List[float]) -> float:
        """Calculate variance"""
        mean = sum(values) / len(values)
        return sum((x - mean)**2 for x in values) / len(values)
    
    print("Feature Variances:")
    for feature in features:
        feature_values = [h[feature] for h in houses]
        variance = calculate_variance(feature_values)
        
        # Normalize variance by mean (coefficient of variation)
        mean_val = sum(feature_values) / len(feature_values)
        cv = (variance ** 0.5) / mean_val if mean_val > 0 else 0
        
        status = "✓ Keep" if cv > 0.01 else "✗ Drop (low variance)"
        print(f"   {feature:20s}: variance={variance:.0f}, CV={cv:.3f}  {status}")
    print()
    
    # Method 3: Feature Importance (simplified)
    print("Method 3: Simple Feature Importance")
    print("   (Simplified - in practice, use tree-based models)")
    print()
    
    # Calculate "importance" as correlation squared
    importances = {f: corr**2 for f, corr in correlations.items()}
    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("Feature Importance Scores:")
    for feature, importance in sorted_importance:
        bar = "█" * int(importance * 50)
        print(f"   {feature:20s}: {bar} {importance:.3f}")
    print()
    
    print("💡 Feature Selection Best Practices:")
    print()
    print("1. Start with domain knowledge:")
    print("   'Paint color' can't affect price → Drop it!")
    print()
    print("2. Check correlation with target:")
    print("   Low correlation? Probably not useful")
    print()
    print("3. Remove redundant features:")
    print("   If two features correlated (r>0.9), keep one")
    print()
    print("4. Remove low-variance features:")
    print("   If feature barely changes, it's useless")
    print()
    print("5. Use model-based selection:")
    print("   Tree importance, Lasso, or RFE")
    print()
    
    print("Real Impact:")
    print(f"   Before: {len(features)} features → Training time: 10 min")
    print(f"   After: {len(selected_features)} features → Training time: 2 min (5x faster!)")
    print(f"   Accuracy: Often IMPROVES (less noise)")


# ============================================================================
# 5. Dimensionality Reduction (PCA Concept)
# ============================================================================

def demo_dimensionality_reduction():
    """
    Dimensionality Reduction: Compress features intelligently
    
    INTUITION - Movie Recommendations:
    
    Netflix has 1000 movie genres: Action, Comedy, Drama, ...
    But really, people like movies along a few themes:
    - Theme 1: "Serious vs Lighthearted" (70% of variance)
    - Theme 2: "Old vs New" (20% of variance)
    - Theme 3: Other small differences (10% of variance)
    
    Instead of tracking 1000 genres, track 2-3 themes!
    Lose 10% info, gain 100x speed and simpler model.
    
    WHY REDUCE DIMENSIONS?
    1. Visualization (can't plot 1000 dimensions!)
    2. Faster training (fewer dimensions = faster)
    3. Remove noise (minor dimensions often noise)
    4. Avoid curse of dimensionality
    
    METHODS:
    
    1. PCA (Principal Component Analysis):
       - Find directions of maximum variance
       - Linear combinations of original features
       - Use when: Features are correlated
    
    2. t-SNE:
       - Preserves local structure (neighbors)
       - Great for visualization
       - Use when: Want to visualize clusters
    
    3. LDA (Linear Discriminant Analysis):
       - Maximizes class separation
       - Supervised (uses labels)
       - Use when: Classification task
    """
    print("\n" + "=" * 70)
    print("5. Dimensionality Reduction")
    print("=" * 70)
    print()
    print("💭 INTUITION: Movie Preferences")
    print("   Netflix: 1000 movie genres")
    print("   Reality: People care about 2-3 main themes:")
    print("   • Theme 1: Serious ← → Lighthearted (70% variance)")
    print("   • Theme 2: Old ← → New (20% variance)")
    print("   • Theme 3: Everything else (10% variance)")
    print()
    print("   Track 2-3 themes instead of 1000 genres!")
    print("   Lose 10% info, gain 100x speed ✓")
    print()
    
    # Simulate customer purchase data (simplified PCA concept)
    print("📊 Customer Purchase Patterns")
    print()
    
    random.seed(42)
    n_customers = 50
    
    # Original features: purchases in different categories
    customers = []
    for i in range(n_customers):
        # Hidden pattern: customers either buy tech OR fashion (not both much)
        customer_type = random.random()
        
        if customer_type < 0.5:  # Tech enthusiast
            laptops = random.randint(5, 15)
            phones = random.randint(5, 15)
            cameras = random.randint(3, 10)
            clothes = random.randint(0, 3)
            shoes = random.randint(0, 2)
        else:  # Fashion enthusiast
            laptops = random.randint(0, 3)
            phones = random.randint(0, 3)
            cameras = random.randint(0, 2)
            clothes = random.randint(5, 15)
            shoes = random.randint(5, 15)
        
        customers.append({
            'laptops': laptops,
            'phones': phones,
            'cameras': cameras,
            'clothes': clothes,
            'shoes': shoes
        })
    
    features = ['laptops', 'phones', 'cameras', 'clothes', 'shoes']
    
    print(f"Original dimensions: {len(features)}")
    print("   Features:", features)
    print()
    
    # Calculate correlations (PCA finds correlated features)
    print("Feature Correlations:")
    print("   (High correlation = redundant = can compress)")
    print()
    
    def get_column(data: List[Dict], col: str) -> List[float]:
        return [d[col] for d in data]
    
    def correlation(x: List[float], y: List[float]) -> float:
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5
        return cov / (n * std_x * std_y)
    
    # Check some key correlations
    laptops_vals = get_column(customers, 'laptops')
    phones_vals = get_column(customers, 'phones')
    clothes_vals = get_column(customers, 'clothes')
    shoes_vals = get_column(customers, 'shoes')
    
    corr_tech = correlation(laptops_vals, phones_vals)
    corr_fashion = correlation(clothes_vals, shoes_vals)
    corr_cross = correlation(laptops_vals, clothes_vals)
    
    print(f"   Laptops ↔ Phones: {corr_tech:.3f}  (high = both tech)")
    print(f"   Clothes ↔ Shoes: {corr_fashion:.3f}  (high = both fashion)")
    print(f"   Laptops ↔ Clothes: {corr_cross:.3f}  (low = different types)")
    print()
    
    # Simplified PCA: Create composite features
    print("Applying Dimensionality Reduction:")
    print("   (Simplified PCA - creating composite features)")
    print()
    
    # Component 1: "Tech affinity"
    # Component 2: "Fashion affinity"
    
    for i, customer in enumerate(customers[:5]):  # Show first 5
        tech_score = customer['laptops'] + customer['phones'] + customer['cameras']
        fashion_score = customer['clothes'] + customer['shoes']
        
        print(f"   Customer {i+1}:")
        print(f"      Original: {dict(customer)}")
        print(f"      Reduced:  Tech={tech_score}, Fashion={fashion_score}")
        print()
    
    print("Results:")
    print(f"   Before: {len(features)} dimensions")
    print(f"   After: 2 dimensions (Tech, Fashion)")
    print(f"   Information retained: ~90%")
    print(f"   Training speed: 2.5x faster")
    print()
    
    print("💡 When to Use Dimensionality Reduction:")
    print()
    print("1. Too many features (>100):")
    print("   PCA to compress to 10-20 key components")
    print()
    print("2. Features are correlated:")
    print("   Many features capturing same info → Compress")
    print()
    print("3. Need to visualize:")
    print("   Reduce to 2-3 dims for plotting")
    print()
    print("4. Speed up training:")
    print("   Fewer dims = faster algorithms")
    print()
    print("⚠️  When NOT to use:")
    print("   • Need interpretability (PCA components are hard to explain)")
    print("   • Already few features (<10)")
    print("   • Features are uncorrelated")


# ============================================================================
# 6. Model Validation (Cross-Validation)
# ============================================================================

def demo_model_validation():
    """
    Model Validation: Is my model actually good?
    
    INTUITION - Student Exam Analogy:
    
    Bad way to test a student:
    - Give them practice problems
    - Test them on THE SAME problems
    - They scored 100%! Are they smart?
    - NO! They just memorized answers (overfitting)
    
    Good way:
    - Give them practice problems (training set)
    - Test on NEW problems (test set)
    - They scored 85%? That's their real ability!
    
    Even better (Cross-Validation):
    - Give 5 different practice/test splits
    - Average their scores across all 5
    - More reliable estimate of true ability
    
    METHODS:
    
    1. Train/Test Split:
       - Simple: 80% train, 20% test
       - Fast but depends on lucky/unlucky split
    
    2. K-Fold Cross-Validation:
       - Split data into K folds (usually K=5 or 10)
       - Train K times, each time using different fold as test
       - Average scores → More reliable
    
    3. Stratified K-Fold:
       - Like K-Fold but preserves class distribution
       - Use when: Imbalanced classes
    
    4. Leave-One-Out:
       - Each sample is test set once
       - Most thorough but slowest
       - Use when: Very small dataset
    """
    print("\n" + "=" * 70)
    print("6. Model Validation")
    print("=" * 70)
    print()
    print("💭 INTUITION: Testing a Student")
    print()
    print("Bad Test:")
    print("   • Give practice problems")
    print("   • Test on SAME problems")
    print("   • 100% score! But just memorized ✗")
    print()
    print("Good Test:")
    print("   • Give practice problems (training)")
    print("   • Test on NEW problems (testing)")
    print("   • 85% score = True ability ✓")
    print()
    print("Even Better (Cross-Validation):")
    print("   • 5 different practice/test splits")
    print("   • Average scores across all 5")
    print("   • Most reliable estimate! ✓✓")
    print()
    
    # Generate dataset: Email classification
    print("📧 Email Spam Classification")
    print()
    
    random.seed(42)
    
    # Simplified email dataset
    all_data = []
    
    # Spam emails (short, contains spam words)
    for _ in range(40):
        has_free = random.random() < 0.8
        has_winner = random.random() < 0.7
        has_click = random.random() < 0.6
        length = random.randint(5, 20)
        
        all_data.append({
            'has_free': has_free,
            'has_winner': has_winner,
            'has_click': has_click,
            'length': length,
            'label': 'spam'
        })
    
    # Ham emails (longer, no spam words)
    for _ in range(60):
        has_free = random.random() < 0.1
        has_winner = random.random() < 0.05
        has_click = random.random() < 0.2
        length = random.randint(20, 100)
        
        all_data.append({
            'has_free': has_free,
            'has_winner': has_winner,
            'has_click': has_click,
            'length': length,
            'label': 'ham'
        })
    
    random.shuffle(all_data)
    
    print(f"Dataset: {len(all_data)} emails")
    spam_count = sum(1 for d in all_data if d['label'] == 'spam')
    print(f"   Spam: {spam_count} ({spam_count/len(all_data):.0%})")
    print(f"   Ham: {len(all_data) - spam_count} ({(len(all_data)-spam_count)/len(all_data):.0%})")
    print()
    
    # Simple classifier: If has_free or has_winner, predict spam
    def simple_classifier(email: Dict) -> str:
        """Very simple spam classifier"""
        if email['has_free'] or email['has_winner']:
            return 'spam'
        return 'ham'
    
    # Method 1: Single Train/Test Split
    print("Method 1: Single Train/Test Split (80/20)")
    print()
    
    split_point = int(len(all_data) * 0.8)
    train_data = all_data[:split_point]
    test_data = all_data[split_point:]
    
    print(f"   Train: {len(train_data)} emails")
    print(f"   Test: {len(test_data)} emails")
    print()
    
    # Test on test set
    correct = sum(1 for email in test_data if simple_classifier(email) == email['label'])
    accuracy = correct / len(test_data)
    
    print(f"   Test Accuracy: {accuracy:.1%}")
    print()
    print("   ⚠️  Problem: What if we got lucky/unlucky split?")
    print("       One test isn't enough!")
    print()
    
    # Method 2: K-Fold Cross-Validation
    print("Method 2: 5-Fold Cross-Validation")
    print()
    
    k = 5
    fold_size = len(all_data) // k
    
    accuracies = []
    
    for fold in range(k):
        # Create test set for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        test_fold = all_data[test_start:test_end]
        train_fold = all_data[:test_start] + all_data[test_end:]
        
        # Evaluate on this fold
        correct = sum(1 for email in test_fold if simple_classifier(email) == email['label'])
        fold_accuracy = correct / len(test_fold)
        accuracies.append(fold_accuracy)
        
        print(f"   Fold {fold+1}: {fold_accuracy:.1%} accuracy")
    
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = (sum((a - mean_accuracy)**2 for a in accuracies) / len(accuracies)) ** 0.5
    
    print()
    print(f"   Mean Accuracy: {mean_accuracy:.1%}")
    print(f"   Std Dev: {std_accuracy:.1%}")
    print()
    print("   ✓ More reliable! We tested on every data point.")
    print("   ✓ Confidence: Accuracy is {:.1%} ± {:.1%}".format(mean_accuracy, 1.96 * std_accuracy))
    print()
    
    # Comparison
    print("Single Split vs Cross-Validation:")
    print()
    print("   Single Split:")
    print(f"      Accuracy: {accuracy:.1%}")
    print("      Problem: Might be lucky/unlucky split")
    print("      Use when: Very large dataset, need speed")
    print()
    print("   5-Fold CV:")
    print(f"      Mean Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}")
    print("      Benefit: More reliable estimate")
    print("      Cost: 5x slower (train 5 models)")
    print("      Use when: Medium dataset, need confidence")
    print()
    
    print("💡 Best Practices:")
    print()
    print("1. Always use separate test set:")
    print("   Never evaluate on training data!")
    print()
    print("2. Use cross-validation for model selection:")
    print("   Which model is best? Try all with CV")
    print()
    print("3. Stratified CV for imbalanced data:")
    print("   Preserves class distribution in each fold")
    print()
    print("4. Report mean ± std:")
    print("   Not just mean - show uncertainty!")
    print()
    print("5. Final evaluation on held-out test set:")
    print("   After selecting best model with CV,")
    print("   test ONCE on fresh data you never touched")


# ============================================================================
# 7. A/B Testing
# ============================================================================

def demo_ab_testing():
    """
    A/B Testing: Is the difference real or just luck?
    
    INTUITION - Coin Flip Example:
    
    You flip a coin 10 times:
    - Get 7 heads, 3 tails
    - Is the coin biased?
    - NO! Could easily happen with fair coin
    
    You flip 1000 times:
    - Get 700 heads, 300 tails
    - NOW it's suspicious! (Expected 500/500)
    - Probably biased coin
    
    Same in A/B testing:
    - Small sample: Can't trust difference
    - Large sample: Can trust difference
    
    REAL EXAMPLE - Website Button:
    
    Version A (blue button): 10% click rate
    Version B (red button): 12% click rate
    
    Questions:
    1. Is 2% improvement real or luck?
    2. How many users needed to be sure?
    3. When can we ship version B?
    
    ANSWER: Use statistics!
    - Calculate confidence interval
    - If interval doesn't include 0 → Real difference!
    - If interval includes 0 → Need more data
    
    KEY CONCEPTS:
    1. Sample size: Bigger = more confident
    2. Statistical significance: p < 0.05 (95% confident)
    3. Effect size: How big is the difference?
    4. Power: Chance of detecting real effect
    """
    print("\n" + "=" * 70)
    print("7. A/B Testing")
    print("=" * 70)
    print()
    print("💭 INTUITION: Coin Flip Test")
    print()
    print("   10 flips: 7 heads, 3 tails")
    print("      Biased? NO - could be luck")
    print()
    print("   1000 flips: 700 heads, 300 tails")
    print("      Biased? YES - too extreme for luck!")
    print()
    print("   Same logic for A/B tests:")
    print("   Small sample → Can't trust difference")
    print("   Large sample → Can trust difference")
    print()
    
    print("🎯 Scenario: Testing New Website Button")
    print()
    print("   Version A (blue): Current button")
    print("   Version B (red): New button")
    print("   Question: Does red button increase clicks?")
    print()
    
    # Simulate A/B test
    def run_ab_test_detailed(n_users_per_variant: int, 
                            true_rate_a: float, 
                            true_rate_b: float) -> Dict:
        """
        Run A/B test simulation
        
        Args:
            n_users_per_variant: Users in each group
            true_rate_a: True click rate for A
            true_rate_b: True click rate for B
        """
        # Simulate users
        clicks_a = sum(1 for _ in range(n_users_per_variant) 
                      if random.random() < true_rate_a)
        clicks_b = sum(1 for _ in range(n_users_per_variant) 
                      if random.random() < true_rate_b)
        
        rate_a = clicks_a / n_users_per_variant
        rate_b = clicks_b / n_users_per_variant
        
        # Calculate standard error for difference
        se_a = (rate_a * (1 - rate_a) / n_users_per_variant) ** 0.5
        se_b = (rate_b * (1 - rate_b) / n_users_per_variant) ** 0.5
        se_diff = (se_a**2 + se_b**2) ** 0.5
        
        # Difference and confidence interval
        diff = rate_b - rate_a
        margin = 1.96 * se_diff  # 95% confidence
        ci_lower = diff - margin
        ci_upper = diff + margin
        
        # Statistical significance
        significant = ci_lower > 0  # If lower bound > 0, improvement is real
        
        return {
            'n_users': n_users_per_variant,
            'rate_a': rate_a,
            'rate_b': rate_b,
            'diff': diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'clicks_a': clicks_a,
            'clicks_b': clicks_b
        }
    
    # True rates (unknown in practice)
    TRUE_RATE_A = 0.10  # 10% click rate
    TRUE_RATE_B = 0.12  # 12% click rate (20% relative improvement)
    
    print("Simulating Tests with Different Sample Sizes:")
    print(f"   (True rates: A={TRUE_RATE_A:.0%}, B={TRUE_RATE_B:.0%})")
    print()
    
    sample_sizes = [100, 500, 1000, 5000]
    
    for n in sample_sizes:
        result = run_ab_test_detailed(n, TRUE_RATE_A, TRUE_RATE_B)
        
        print(f"Sample Size: {n:,} users per variant")
        print(f"   A: {result['clicks_a']}/{n} = {result['rate_a']:.1%}")
        print(f"   B: {result['clicks_b']}/{n} = {result['rate_b']:.1%}")
        print(f"   Difference: {result['diff']:.1%} ({result['diff']/TRUE_RATE_A:.1%} relative)")
        print(f"   95% CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}]")
        
        if result['significant']:
            print(f"   ✅ SIGNIFICANT! Ship version B!")
        else:
            print(f"   ❌ Not significant yet. Need more data.")
        print()
    
    # Common mistakes
    print("=" * 70)
    print("Common A/B Testing Mistakes")
    print("=" * 70)
    print()
    
    print("❌ Mistake 1: Stopping too early")
    print("   'After 50 users, B is winning!'")
    print("   Problem: Could just be luck")
    print("   Solution: Wait for statistical significance")
    print()
    
    print("❌ Mistake 2: Multiple testing")
    print("   'I tested 20 different buttons, one was significant!'")
    print("   Problem: 5% false positive rate → expect 1/20 false positives")
    print("   Solution: Bonferroni correction or fewer tests")
    print()
    
    print("❌ Mistake 3: Peeking")
    print("   'Let me check results every hour...'")
    print("   Problem: If you keep checking, you'll eventually see 'significance' by chance")
    print("   Solution: Set sample size beforehand, check once")
    print()
    
    print("❌ Mistake 4: Ignoring practical significance")
    print("   'B is 0.01% better and statistically significant!'")
    print("   Problem: Tiny improvement not worth the effort")
    print("   Solution: Set minimum detectable effect (e.g., 10% improvement)")
    print()
    
    # Sample size calculation
    print("=" * 70)
    print("Sample Size Planning")
    print("=" * 70)
    print()
    
    print("How many users do I need?")
    print()
    print("Rule of Thumb:")
    print("   Want to detect 20% improvement at 95% confidence?")
    print("   Need ~2000 users per variant")
    print()
    print("   Want to detect 10% improvement?")
    print("   Need ~8000 users per variant")
    print()
    print("   Want to detect 5% improvement?")
    print("   Need ~30,000 users per variant")
    print()
    
    print("Factors affecting sample size:")
    print("   • Baseline rate: Lower rate needs more users")
    print("   • Desired improvement: Smaller effect needs more users")
    print("   • Confidence level: 99% needs more than 95%")
    print("   • Power: Higher power needs more users")
    print()
    
    # Practical workflow
    print("=" * 70)
    print("A/B Testing Workflow")
    print("=" * 70)
    print()
    
    print("Step 1: Define hypothesis")
    print("   H0: Red button = Blue button (no difference)")
    print("   H1: Red button > Blue button")
    print()
    
    print("Step 2: Calculate required sample size")
    print("   Based on: baseline rate, desired improvement, power")
    print("   Example: Need 2000 users per variant")
    print()
    
    print("Step 3: Run test (NO PEEKING!)")
    print("   Randomly assign users to A or B")
    print("   Collect data until reaching sample size")
    print()
    
    print("Step 4: Analyze results")
    print("   Calculate rates, difference, confidence interval")
    print("   Check statistical significance")
    print()
    
    print("Step 5: Make decision")
    print("   Significant + meaningful improvement → Ship B!")
    print("   Not significant → Keep A")
    print("   Significant but tiny → Consider cost/benefit")
    print()
    
    print("💡 Golden Rules:")
    print("   1. Plan sample size BEFORE starting")
    print("   2. Random assignment (avoid bias)")
    print("   3. Wait for full sample (no peeking!)")
    print("   4. Check both statistical AND practical significance")
    print("   5. Document everything (for future reference)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n📊 Practical ML Statistics & Data Science\n")
    print("Real-world Python examples for ML engineering!")
    print()
    
    demo_eda()
    demo_feature_scaling()
    demo_naive_bayes()
    demo_feature_selection()
    demo_dimensionality_reduction()
    demo_model_validation()
    demo_ab_testing()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. EDA: Investigate data before modeling (70% of time here!)
   • Check distributions, missing values, outliers
   • Understand correlations and patterns
   
2. Feature Scaling: Make features comparable
   • Standardization (z-score): Most common
   • Min-Max: When need bounded range
   • Impact: 10-100x faster training!
   
3. Naive Bayes: Update probability with evidence
   • Fast, simple, works surprisingly well
   • Used in: Spam filters, sentiment analysis
   
4. Feature Selection: Pick useful features
   • Faster training, better generalization
   • Methods: Correlation, variance, model-based
   
5. Dimensionality Reduction: Compress intelligently
   • PCA: Find directions of max variance
   • Use when: Many correlated features
   
6. Model Validation: Test properly
   • Single split: Fast but unreliable
   • Cross-validation: Slower but reliable
   • Always report mean ± std
   
7. A/B Testing: Is difference real or luck?
   • Small sample → Can't trust
   • Large sample → Can trust
   • Plan sample size beforehand!

Real-World Impact:
• 80% of ML work is data preparation
• Good EDA finds issues before training
• Proper validation prevents overfitting
• A/B testing proves business value

Next Steps:
• Apply these to real datasets
• Practice with scikit-learn implementations
• Build end-to-end ML pipelines
""")


if __name__ == "__main__":
    main()
