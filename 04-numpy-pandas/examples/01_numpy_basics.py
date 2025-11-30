"""
NumPy Basics for ML/AI Engineering

Real-world examples showing why NumPy is 10-100x faster than Python lists.
Focus: Practical operations you'll use in data processing and ML.

Install: poetry add numpy
Run: poetry run python 04-numpy-pandas/examples/01_numpy_basics.py
"""

import numpy as np
import time
from typing import List


# ============================================================================
# 1. Why NumPy? Speed Comparison
# ============================================================================

def demo_speed_comparison():
    """
    INTUITION - Processing Server Logs:
    
    You have 1 million API response times to analyze.
    Python list: Process one at a time (slow!)
    NumPy array: Process all at once (vectorized, fast!)
    
    Real Impact:
    - Python list: 1 second
    - NumPy: 0.01 seconds
    - 100x speedup!
    """
    print("=" * 70)
    print("1. Why NumPy? Speed Comparison")
    print("=" * 70)
    print()
    print("💡 INTUITION: Processing 1 Million API Response Times")
    print("   Python list: Loop through each one (slow)")
    print("   NumPy array: Process all at once (vectorized)")
    print()
    
    # Create large dataset
    size = 1_000_000
    
    # Python list
    python_list = list(range(size))
    
    # NumPy array
    numpy_array = np.arange(size)
    
    print(f"Dataset: {size:,} numbers")
    print()
    
    # Task: Square each number
    print("Task: Square each number")
    print()
    
    # Python list approach
    start = time.time()
    python_squared = [x ** 2 for x in python_list]
    python_time = time.time() - start
    
    # NumPy approach
    start = time.time()
    numpy_squared = numpy_array ** 2
    numpy_time = time.time() - start
    
    print(f"Python list: {python_time:.4f} seconds")
    print(f"NumPy array: {numpy_time:.4f} seconds")
    print(f"Speedup: {python_time / numpy_time:.1f}x faster!")
    print()
    
    print("🚀 Real-World Impact:")
    print("   - Image processing: Process millions of pixels instantly")
    print("   - ML training: Compute gradients for millions of weights")
    print("   - Data analysis: Aggregate millions of records in <1 second")
    print()
    
    # Memory usage comparison
    print("Memory Usage:")
    python_size = python_list.__sizeof__() + sum(x.__sizeof__() for x in python_list[:100]) * (size // 100)
    numpy_size = numpy_array.nbytes
    
    print(f"Python list: ~{python_size / 1_000_000:.1f} MB")
    print(f"NumPy array: {numpy_size / 1_000_000:.1f} MB")
    print(f"NumPy uses {python_size / numpy_size:.1f}x less memory!")


# ============================================================================
# 2. Creating Arrays - Your Data as NumPy
# ============================================================================

def demo_creating_arrays():
    """
    INTUITION - Different Ways to Load Data:
    
    - From list: Manual data entry or small datasets
    - From range: Generate sequences (timestamps, IDs)
    - Zeros/Ones: Initialize arrays (like empty spreadsheet)
    - Random: Simulate data, initialize neural network weights
    """
    print("\n" + "=" * 70)
    print("2. Creating Arrays")
    print("=" * 70)
    print()
    print("💡 INTUITION: Different Data Sources")
    print("   From list: Manual entry, small datasets")
    print("   Range: Sequential data (timestamps, IDs)")
    print("   Zeros: Empty placeholder (like blank spreadsheet)")
    print("   Random: Simulations, ML weight initialization")
    print()
    
    # From Python list
    prices = [29.99, 39.99, 19.99, 49.99, 24.99]
    prices_array = np.array(prices)
    
    print("📊 From list (Product Prices):")
    print(f"   {prices_array}")
    print(f"   Type: {type(prices_array)}")
    print(f"   Shape: {prices_array.shape}")
    print()
    
    # Sequential data
    user_ids = np.arange(1000, 1010)
    print("🔢 Sequential (User IDs):")
    print(f"   {user_ids}")
    print()
    
    # Evenly spaced (like time series)
    timestamps = np.linspace(0, 10, 5)  # 5 points from 0 to 10
    print("⏰ Evenly spaced (Timestamps, 0-10 seconds):")
    print(f"   {timestamps}")
    print()
    
    # Zeros (initialize empty data)
    empty_data = np.zeros(5)
    print("📋 Zeros (Empty placeholders):")
    print(f"   {empty_data}")
    print()
    
    # Ones (baseline, binary masks)
    all_active = np.ones(5)
    print("✓ Ones (All active users):")
    print(f"   {all_active}")
    print()
    
    # Random data (simulations, weight initialization)
    random_data = np.random.random(5)
    print("🎲 Random (Simulated click rates):")
    print(f"   {random_data}")
    print()
    
    # Random integers (dice rolls, user choices)
    dice_rolls = np.random.randint(1, 7, size=10)
    print("🎲 Random integers (10 dice rolls):")
    print(f"   {dice_rolls}")


# ============================================================================
# 3. Array Indexing and Slicing - Accessing Data
# ============================================================================

def demo_indexing_slicing():
    """
    INTUITION - Working with Time Series Data:
    
    You have hourly sales data for a week (168 hours).
    - Index: Get specific hour
    - Slice: Get range (like "Monday's data")
    - Boolean mask: Get all hours with sales > $1000
    """
    print("\n" + "=" * 70)
    print("3. Indexing and Slicing")
    print("=" * 70)
    print()
    print("💡 INTUITION: Sales Data for a Week")
    print("   Index: Get specific hour (hour 5)")
    print("   Slice: Get range (first 24 hours = Monday)")
    print("   Boolean mask: Find all hours with sales > $1000")
    print()
    
    # Create sample sales data (168 hours = 1 week)
    np.random.seed(42)
    sales = np.random.randint(500, 2000, size=168)
    
    print(f"📊 Sales data: {len(sales)} hours")
    print(f"   First 10 hours: {sales[:10]}")
    print()
    
    # Single element
    hour_5 = sales[5]
    print(f"Hour 5 sales: ${hour_5}")
    print()
    
    # Slice (first 24 hours = Monday)
    monday = sales[:24]
    print(f"Monday (first 24h) average: ${monday.mean():.2f}")
    print()
    
    # Last 24 hours (Sunday)
    sunday = sales[-24:]
    print(f"Sunday (last 24h) average: ${sunday.mean():.2f}")
    print()
    
    # Every 24th hour (same hour each day)
    same_hour_each_day = sales[::24]
    print(f"Hour 0 each day (7 days): {same_hour_each_day}")
    print()
    
    # Boolean indexing (find peak hours)
    threshold = 1500
    peak_hours = sales > threshold
    peak_sales = sales[peak_hours]
    
    print(f"Peak hours (sales > ${threshold}):")
    print(f"   Count: {len(peak_sales)} hours")
    print(f"   Average during peaks: ${peak_sales.mean():.2f}")
    print(f"   Peak indices: {np.where(peak_hours)[0][:5]}... (showing first 5)")
    print()
    
    # Multiple conditions (peak hours on weekdays)
    # Hours 0-119 = Mon-Fri (5 days * 24 hours)
    weekday_mask = np.arange(len(sales)) < 120
    weekday_peaks = sales[(peak_hours) & (weekday_mask)]
    print(f"Weekday peaks: {len(weekday_peaks)} hours")


# ============================================================================
# 4. Array Operations - Vectorized Math
# ============================================================================

def demo_array_operations():
    """
    INTUITION - Processing E-commerce Orders:
    
    You have arrays of:
    - Prices: [10, 20, 15, 30]
    - Quantities: [2, 1, 3, 1]
    - Discount rates: [0.1, 0, 0.15, 0.2]
    
    Calculate: Total revenue, apply discounts, find statistics
    """
    print("\n" + "=" * 70)
    print("4. Array Operations - Vectorized Math")
    print("=" * 70)
    print()
    print("💡 INTUITION: Processing E-commerce Orders")
    print("   Calculate revenue for 10,000 orders instantly")
    print("   Apply discounts, taxes, shipping all at once")
    print()
    
    # Sample order data
    prices = np.array([10.00, 20.00, 15.00, 30.00, 25.00])
    quantities = np.array([2, 1, 3, 1, 2])
    discount_rates = np.array([0.1, 0.0, 0.15, 0.2, 0.05])
    
    print("📦 Orders:")
    print(f"   Prices:     {prices}")
    print(f"   Quantities: {quantities}")
    print(f"   Discounts:  {discount_rates}")
    print()
    
    # Calculate subtotals (element-wise multiplication)
    subtotals = prices * quantities
    print(f"Subtotals: {subtotals}")
    print()
    
    # Apply discounts
    discount_amounts = subtotals * discount_rates
    finals = subtotals - discount_amounts
    
    print(f"Discounts:  {discount_amounts}")
    print(f"Finals:     {finals}")
    print()
    
    # Statistics
    print("📊 Order Statistics:")
    print(f"   Total revenue: ${finals.sum():.2f}")
    print(f"   Average order: ${finals.mean():.2f}")
    print(f"   Largest order: ${finals.max():.2f}")
    print(f"   Smallest order: ${finals.min():.2f}")
    print(f"   Std deviation: ${finals.std():.2f}")
    print()
    
    # Real-world: Scale to 10,000 orders
    print("🚀 Scaling to Production (10,000 orders):")
    large_prices = np.random.uniform(10, 100, size=10_000)
    large_quantities = np.random.randint(1, 5, size=10_000)
    large_discounts = np.random.uniform(0, 0.3, size=10_000)
    
    start = time.time()
    large_subtotals = large_prices * large_quantities
    large_discount_amounts = large_subtotals * large_discounts
    large_finals = large_subtotals - large_discount_amounts
    total_revenue = large_finals.sum()
    elapsed = time.time() - start
    
    print(f"   Processed 10,000 orders in {elapsed*1000:.2f}ms")
    print(f"   Total revenue: ${total_revenue:,.2f}")
    print(f"   Average order: ${large_finals.mean():.2f}")


# ============================================================================
# 5. Broadcasting - Smart Array Operations
# ============================================================================

def demo_broadcasting():
    """
    INTUITION - Applying Tax to Multi-Region Sales:
    
    You have sales across 3 products and 4 regions.
    Each region has different tax rate.
    
    Broadcasting: Automatically expands dimensions to match!
    No loops needed, NumPy handles it.
    """
    print("\n" + "=" * 70)
    print("5. Broadcasting - Smart Array Operations")
    print("=" * 70)
    print()
    print("💡 INTUITION: Multi-Region Sales with Different Taxes")
    print("   Sales: 3 products × 4 regions")
    print("   Tax rates: 4 regions (different rates)")
    print("   Broadcasting: Applies correct tax to each region automatically!")
    print()
    
    # Sales data: 3 products × 4 regions
    sales = np.array([
        [1000, 1200, 800, 1500],   # Product A across 4 regions
        [2000, 1800, 2200, 1900],  # Product B
        [1500, 1600, 1400, 1700],  # Product C
    ])
    
    # Tax rates per region (1D array)
    tax_rates = np.array([0.08, 0.10, 0.06, 0.09])  # 8%, 10%, 6%, 9%
    
    print("📊 Sales Matrix (Products × Regions):")
    print(sales)
    print()
    print(f"Tax Rates per Region: {tax_rates}")
    print()
    
    # Broadcasting: tax_rates automatically expands to match sales shape
    taxes = sales * tax_rates
    total_with_tax = sales + taxes
    
    print("Broadcasting in action:")
    print("   sales (3×4) * tax_rates (4,)")
    print("   NumPy auto-expands tax_rates to (3×4)")
    print()
    print("Taxes collected:")
    print(taxes)
    print()
    print("Total with tax:")
    print(total_with_tax)
    print()
    
    # Summary by region
    print("Summary by Region:")
    for i, region in enumerate(['West', 'East', 'South', 'North']):
        region_sales = sales[:, i].sum()
        region_tax = taxes[:, i].sum()
        print(f"   {region:6s}: Sales ${region_sales:,}, Tax ${region_tax:.2f} ({tax_rates[i]:.0%})")
    print()
    
    # Another example: Normalize features
    print("🔧 Another Example: Feature Normalization")
    print("   Data: 1000 users × 3 features (age, income, score)")
    print()
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(1000, 3) * np.array([10, 20000, 100]) + np.array([30, 50000, 700])
    
    print("Before normalization (first 5 users):")
    print(data[:5])
    print()
    
    # Standardize: (x - mean) / std for each feature
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    
    # Broadcasting: means and stds (shape 3,) work with data (1000, 3)
    normalized = (data - means) / stds
    
    print("After normalization (first 5 users):")
    print(normalized[:5])
    print()
    print(f"New means (should be ~0): {normalized.mean(axis=0)}")
    print(f"New stds (should be ~1): {normalized.std(axis=0)}")


# ============================================================================
# 6. Aggregations - Summary Statistics
# ============================================================================

def demo_aggregations():
    """
    INTUITION - Analyzing Website Traffic:
    
    You have page views: 30 days × 24 hours = 720 data points
    Questions:
    - Total traffic?
    - Daily average?
    - Peak hour?
    - Weekend vs weekday?
    """
    print("\n" + "=" * 70)
    print("6. Aggregations - Summary Statistics")
    print("=" * 70)
    print()
    print("💡 INTUITION: Website Traffic Analysis")
    print("   Data: 30 days × 24 hours = 720 data points")
    print("   Questions: Total, daily avg, peak hour, trends")
    print()
    
    # Generate realistic traffic data (30 days × 24 hours)
    np.random.seed(42)
    
    # Base traffic with daily and hourly patterns
    days = 30
    hours_per_day = 24
    base_traffic = 1000
    
    # Create hourly pattern (low at night, high during day)
    hourly_pattern = np.array([
        0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2,  # Midnight to 8am
        1.5, 1.8, 2.0, 2.0, 1.8, 1.6, 1.4, 1.5,  # 8am to 4pm
        1.8, 2.0, 1.8, 1.5, 1.2, 0.9, 0.6, 0.4   # 4pm to midnight
    ])
    
    traffic = np.zeros((days, hours_per_day))
    for day in range(days):
        daily_multiplier = 1.0 + 0.1 * np.sin(day / 7 * np.pi)  # Weekly pattern
        noise = np.random.normal(0, 0.1, hours_per_day)
        traffic[day] = base_traffic * hourly_pattern * daily_multiplier * (1 + noise)
    
    print(f"📊 Traffic Data: {traffic.shape} (30 days × 24 hours)")
    print()
    
    # Total traffic
    total = traffic.sum()
    print(f"Total page views (30 days): {total:,.0f}")
    print()
    
    # Daily statistics
    daily_totals = traffic.sum(axis=1)  # Sum across hours
    print(f"Daily Statistics:")
    print(f"   Average:  {daily_totals.mean():,.0f} views/day")
    print(f"   Best day: {daily_totals.max():,.0f} views")
    print(f"   Worst day: {daily_totals.min():,.0f} views")
    print()
    
    # Hourly statistics (average across all days)
    hourly_avg = traffic.mean(axis=0)  # Mean across days
    peak_hour = hourly_avg.argmax()
    print(f"Hourly Patterns:")
    print(f"   Peak hour: {peak_hour}:00 ({hourly_avg[peak_hour]:,.0f} avg views)")
    print(f"   Quiet hour: {hourly_avg.argmin()}:00 ({hourly_avg.min():,.0f} avg views)")
    print()
    
    # Show hourly pattern
    print("Average views by hour of day:")
    for hour in [0, 6, 12, 18]:
        print(f"   {hour:2d}:00 - {hourly_avg[hour]:6,.0f} views")
    print()
    
    # Percentiles
    print("Traffic Distribution:")
    print(f"   25th percentile: {np.percentile(traffic, 25):,.0f} views")
    print(f"   50th percentile (median): {np.percentile(traffic, 50):,.0f} views")
    print(f"   75th percentile: {np.percentile(traffic, 75):,.0f} views")
    print(f"   95th percentile: {np.percentile(traffic, 95):,.0f} views")


# ============================================================================
# 7. Real-World: Image Processing
# ============================================================================

def demo_image_processing():
    """
    INTUITION - Processing Product Images:
    
    Image = 3D NumPy array (height, width, RGB channels)
    Operations:
    - Resize
    - Convert to grayscale
    - Apply filters
    - Normalize for ML
    """
    print("\n" + "=" * 70)
    print("7. Real-World: Image Processing")
    print("=" * 70)
    print()
    print("💡 INTUITION: Product Image for ML Model")
    print("   Image = 3D array (height × width × 3 RGB channels)")
    print("   Operations: Resize, grayscale, normalize for ML")
    print()
    
    # Simulate a color image (100×100 pixels, 3 color channels)
    np.random.seed(42)
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    
    print(f"📷 Original Image: {image.shape} (height × width × channels)")
    print(f"   Data type: {image.dtype}")
    print(f"   Value range: {image.min()} to {image.max()}")
    print(f"   Memory: {image.nbytes / 1024:.2f} KB")
    print()
    
    # Convert to grayscale (weighted average of RGB)
    # Human eye formula: 0.299*R + 0.587*G + 0.114*B
    weights = np.array([0.299, 0.587, 0.114])
    grayscale = np.dot(image, weights)
    
    print(f"Grayscale Image: {grayscale.shape}")
    print(f"   Value range: {grayscale.min():.1f} to {grayscale.max():.1f}")
    print()
    
    # Normalize for ML (scale to 0-1)
    normalized = image / 255.0
    print(f"Normalized for ML: {normalized.shape}")
    print(f"   Value range: {normalized.min():.3f} to {normalized.max():.3f}")
    print()
    
    # Resize by downsampling (take every other pixel)
    resized = image[::2, ::2, :]  # Half the size
    print(f"Resized Image: {resized.shape}")
    print(f"   Size reduction: {image.nbytes / resized.nbytes:.1f}x smaller")
    print()
    
    # Apply simple filter (brighten)
    brightened = np.clip(image * 1.3, 0, 255).astype(np.uint8)
    print(f"Brightened: Multiplied by 1.3, clipped to 0-255")
    print()
    
    # Real ML preprocessing pipeline
    print("🔧 ML Preprocessing Pipeline:")
    print("   1. Load image as NumPy array")
    print("   2. Resize to model input size (e.g., 224×224)")
    print("   3. Normalize to [0, 1] or [-1, 1]")
    print("   4. Add batch dimension (1, 224, 224, 3)")
    print("   5. Feed to model!")
    print()
    
    # Demonstrate batch processing
    batch_size = 32
    print(f"Processing batch of {batch_size} images:")
    batch = np.random.randint(0, 256, size=(batch_size, 224, 224, 3), dtype=np.uint8)
    normalized_batch = batch / 255.0
    
    start = time.time()
    # Simulate some processing
    _ = normalized_batch.mean(axis=(1, 2, 3))  # Average color per image
    elapsed = time.time() - start
    
    print(f"   Shape: {batch.shape}")
    print(f"   Processed in {elapsed*1000:.2f}ms")
    print(f"   That's {elapsed*1000/batch_size:.2f}ms per image!")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🔢 NumPy Basics for ML/AI Engineering\n")
    print("Focus: Real-world operations, speed, and vectorization!")
    print()
    
    demo_speed_comparison()
    demo_creating_arrays()
    demo_indexing_slicing()
    demo_array_operations()
    demo_broadcasting()
    demo_aggregations()
    demo_image_processing()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Speed: NumPy is 10-100x faster than Python lists
2. Memory: Uses less memory (arrays vs lists of objects)
3. Vectorization: Operations on entire arrays at once
4. Broadcasting: Smart dimension matching (no loops!)
5. Rich Operations: Math, stats, aggregations built-in

Why NumPy in ML:
- Fast data preprocessing
- Efficient matrix operations (neural networks)
- Image/signal processing
- Statistical computations
- Memory-efficient large datasets

Next: 02_numpy_advanced.py - Advanced operations for ML!
""")


if __name__ == "__main__":
    main()
