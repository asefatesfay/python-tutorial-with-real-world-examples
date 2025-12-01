"""
Simple CNN for Image Classification

Build a Convolutional Neural Network from scratch (conceptually).
Learn why CNNs dominate computer vision.

Install: poetry add torch torchvision pillow
Run: poetry run python 07-deep-learning/fundamentals/02_simple_cnn.py
"""

import random
import math
from typing import List, Tuple


# ============================================================================
# 1. Why CNNs? The Problem with Regular Neural Networks
# ============================================================================

def demo_why_cnns():
    """
    Why Convolutional Neural Networks for images?
    
    INTUITION - The Photo Detective Problem:
    
    Regular Neural Network (Fully Connected):
    "Look at every single pixel individually, find the cat"
    
    Problem with 224×224 color image:
    - 224 × 224 × 3 = 150,528 pixels
    - First layer needs 150,528 × 1000 = 150 MILLION weights!
    - Can't learn patterns (edge of cat's ear could be anywhere)
    - Overfits like crazy
    
    CNN Approach:
    "Scan image with small filters, detect patterns locally"
    
    Benefits:
    - Finds edges regardless of position (translation invariant)
    - Dramatically fewer parameters (efficient!)
    - Learns hierarchy: edges → shapes → objects
    
    Real Analogy - Finding Waldo:
    
    Bad approach (fully connected):
    "Memorize every possible position of Waldo"
    → Need to see Waldo in EVERY location
    
    Good approach (CNN):
    "Learn what Waldo looks like (red/white stripes, hat, glasses)"
    → Find Waldo anywhere in the image!
    
    WHY CNNS DOMINATE COMPUTER VISION:
    
    1. Translation Invariance:
       Cat in top-left or bottom-right? Same detection!
       
    2. Parameter Sharing:
       One filter scans entire image (efficient)
       
    3. Hierarchical Learning:
       Layer 1: Edges (horizontal, vertical, diagonal)
       Layer 2: Textures (fur, scales, feathers)
       Layer 3: Parts (ears, eyes, nose)
       Layer 4: Objects (cat, dog, bird)
    
    Real Impact:
    - ImageNet 2012: AlexNet (CNN) wins by huge margin
    - Before: 75% accuracy, After: 85%+
    - Started deep learning revolution!
    """
    print("=" * 70)
    print("1. Why CNNs for Images?")
    print("=" * 70)
    print()
    print("💭 INTUITION: Finding Waldo")
    print()
    print("   ❌ Bad Approach (Fully Connected):")
    print("      'Memorize Waldo in every possible position'")
    print("      • Need 1000s of training examples")
    print("      • Waldo at new position? Model fails!")
    print("      • Parameters: 150 MILLION for 224×224 image")
    print()
    print("   ✅ Good Approach (CNN):")
    print("      'Learn what Waldo looks like'")
    print("      • Red/white stripes, hat, glasses")
    print("      • Scan image, find pattern anywhere")
    print("      • Parameters: 10,000 (99% reduction!)")
    print()
    
    print("📊 The Numbers:")
    print()
    print("   Image: 224×224 RGB (224 × 224 × 3 = 150,528 pixels)")
    print()
    print("   Fully Connected Network:")
    print("   • Input: 150,528 neurons")
    print("   • Hidden: 1,000 neurons")
    print("   • Parameters: 150,528 × 1,000 = 150,528,000")
    print("   • Problems:")
    print("     - Overfits (too many parameters)")
    print("     - Can't generalize to new positions")
    print("     - Slow to train")
    print()
    print("   CNN:")
    print("   • Filters: 3×3 size, 32 filters")
    print("   • Parameters per filter: 3 × 3 × 3 = 27")
    print("   • Total: 32 × 27 = 864 parameters")
    print("   • Benefits:")
    print("     - Generalizes (finds patterns anywhere)")
    print("     - Fast to train")
    print("     - State-of-the-art results")
    print()
    
    print("🎯 Real Example: Cat Detection")
    print()
    print("   Fully Connected:")
    print("   • Learns: 'Cat has pixels at positions (12,45), (13,45)...'")
    print("   • New cat photo with cat in different spot? FAILS")
    print()
    print("   CNN:")
    print("   • Learns: 'Cat has triangular ears, whiskers, fur texture'")
    print("   • New cat photo in any position? SUCCESS ✓")
    print()
    
    print("🧠 How CNNs Learn Hierarchy:")
    print()
    print("   Layer 1 (Edges):")
    print("   • Horizontal edge detector: |‾‾‾|")
    print("   • Vertical edge detector:   | | |")
    print("   • Diagonal edge detector:   |／|")
    print()
    print("   Layer 2 (Textures):")
    print("   • Combine edges → Fur pattern")
    print("   • Combine edges → Scale pattern")
    print("   • Combine edges → Feather pattern")
    print()
    print("   Layer 3 (Parts):")
    print("   • Combine textures → Cat ear")
    print("   • Combine textures → Dog nose")
    print("   • Combine textures → Bird beak")
    print()
    print("   Layer 4 (Objects):")
    print("   • Combine parts → Cat!")
    print("   • Combine parts → Dog!")
    print("   • Combine parts → Bird!")
    print()
    
    print("💡 Key Innovation: Convolutional Filters")
    print()
    print("   Filter = Small pattern detector (e.g., 3×3)")
    print("   Slides across image, looking for pattern")
    print()
    print("   Example: Vertical Edge Detector")
    print("   Filter:    Image section:    Response:")
    print("   [-1 0 1]   [0 0 255]        High!")
    print("   [-1 0 1] * [0 0 255]  →     (Found")
    print("   [-1 0 1]   [0 0 255]         edge!)")
    print()
    print("   Same filter scans ENTIRE image")
    print("   Finds vertical edges everywhere (efficient!)")


# ============================================================================
# 2. Convolution Operation
# ============================================================================

def demo_convolution():
    """
    Convolution: Sliding a filter over an image
    
    INTUITION - The Scanner Metaphor:
    
    You're proofreading a document for typos.
    
    Bad way:
    Read entire page at once (overwhelming!)
    
    Good way:
    Use a reading guide (ruler under each line)
    Scan line by line, check for errors
    
    Convolution is like the reading guide:
    - Small filter (3×3) scans image
    - Checks each local region
    - Detects patterns (edges, corners, textures)
    
    How It Works:
    
    1. Place filter on top-left of image
    2. Multiply filter values × image values
    3. Sum up results → Single output value
    4. Slide filter right, repeat
    5. When reach end of row, go down and repeat
    
    Example: Edge Detection
    
    Image (grayscale):
    [0 0 0 255 255]  ← Left side dark, right side bright
    [0 0 0 255 255]
    [0 0 0 255 255]
    
    Vertical Edge Filter:
    [-1  0  1]
    [-1  0  1]
    [-1  0  1]
    
    Slide filter across image:
    - Over dark region: 0 (no edge)
    - Over edge: 765 (strong edge!)
    - Over bright region: 0 (no edge)
    
    Output shows WHERE the edges are!
    """
    print("\n" + "=" * 70)
    print("2. Convolution Operation")
    print("=" * 70)
    print()
    print("💭 INTUITION: Proofreading with a Reading Guide")
    print()
    print("   ❌ Read entire page at once:")
    print("      Too much information, miss errors")
    print()
    print("   ✅ Use reading guide (ruler):")
    print("      Focus on one line at a time")
    print("      Scan systematically, catch every typo")
    print()
    print("   Convolution = Reading guide for images!")
    print("   Small filter scans image, detects patterns")
    print()
    
    # Simple edge detection example
    print("🎯 Example: Vertical Edge Detection")
    print()
    
    # Simple 5x5 image (dark left, bright right)
    image = [
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255]
    ]
    
    # 3x3 vertical edge filter
    filter_vert = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
    
    print("   Input Image (5×5):")
    print("   (0=black, 255=white)")
    for row in image:
        print(f"   {row}")
    print()
    
    print("   Vertical Edge Filter (3×3):")
    for row in filter_vert:
        print(f"   {row}")
    print()
    
    # Perform convolution (simplified - one position)
    print("   Convolution Step-by-Step (at position row=1, col=1):")
    print()
    
    # Extract 3x3 region
    region = []
    for i in range(3):
        region.append(image[i][0:3])
    
    print("   Image region (3×3):")
    for row in region:
        print(f"     {row}")
    print()
    
    # Element-wise multiply and sum
    result = 0
    print("   Multiply filter × image:")
    for i in range(3):
        for j in range(3):
            product = filter_vert[i][j] * region[i][j]
            result += product
            print(f"     {filter_vert[i][j]:2d} × {region[i][j]:3d} = {product:4d}")
    
    print(f"\n   Sum: {result}")
    print(f"   This is the output value at position (1,1)")
    print()
    
    # Perform full convolution
    def convolve_2d(image: List[List[int]], kernel: List[List[int]]) -> List[List[int]]:
        """Simple 2D convolution (no padding)."""
        img_h, img_w = len(image), len(image[0])
        ker_h, ker_w = len(kernel), len(kernel[0])
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        
        output = []
        for i in range(out_h):
            row = []
            for j in range(out_w):
                # Extract region
                value = 0
                for ki in range(ker_h):
                    for kj in range(ker_w):
                        value += image[i + ki][j + kj] * kernel[ki][kj]
                row.append(value)
            output.append(row)
        return output
    
    output = convolve_2d(image, filter_vert)
    
    print("   Full Convolution Output (3×3):")
    for row in output:
        print(f"   {row}")
    print()
    
    print("   💡 Interpretation:")
    print("   • High values (765): Strong vertical edge detected!")
    print("   • Low values (0, -255): No edge")
    print("   • Filter found the edge where dark meets bright")
    print()
    
    print("🔄 What Happens When Filter Slides:")
    print()
    print("   Position 1 (left side):")
    print("   Filter sees: [0, 0, 0] → No edge (output ≈ 0)")
    print()
    print("   Position 2 (at edge):")
    print("   Filter sees: [0, 0, 255] → Edge! (output = 765)")
    print()
    print("   Position 3 (right side):")
    print("   Filter sees: [255, 255, 255] → No edge (output ≈ 0)")
    print()
    
    print("💡 Multiple Filters = Multiple Feature Maps:")
    print()
    print("   Vertical edge filter → Detects | edges")
    print("   Horizontal edge filter → Detects ‾ edges")
    print("   Diagonal edge filter → Detects / edges")
    print()
    print("   Each filter creates one feature map")
    print("   32 filters → 32 feature maps (rich representation!)")


# ============================================================================
# 3. CNN Architecture
# ============================================================================

def demo_cnn_architecture():
    """
    CNN Architecture: Stack of Conv, Pool, and FC layers
    
    INTUITION - The Manufacturing Assembly Line:
    
    Raw materials (image) → Final product (classification)
    
    Station 1: Convolutional Layer
    "Extract features (edges, textures)"
    Like quality inspector checking for defects
    
    Station 2: Activation (ReLU)
    "Keep important features, discard noise"
    Like filter removing bad parts
    
    Station 3: Pooling
    "Reduce size, keep essential info"
    Like summarizer: 4 items → 1 representative
    
    Station 4-6: Repeat
    "Build higher-level features"
    
    Final Station: Fully Connected
    "Make decision based on all features"
    Like manager reviewing reports, making final call
    
    Classic CNN (e.g., LeNet, AlexNet):
    
    Input Image (224×224×3)
        ↓
    Conv Layer 1 (3×3 filters, 32 channels)
    → Detects edges, basic patterns
    → Output: 222×222×32
        ↓
    ReLU Activation
    → Remove negative values
        ↓
    MaxPool (2×2)
    → Reduce size by half
    → Output: 111×111×32
        ↓
    Conv Layer 2 (3×3 filters, 64 channels)
    → Detects textures, shapes
    → Output: 109×109×64
        ↓
    ReLU + MaxPool
    → Output: 54×54×64
        ↓
    Conv Layer 3 (3×3 filters, 128 channels)
    → Detects parts (ears, eyes)
    → Output: 52×52×128
        ↓
    ReLU + MaxPool
    → Output: 26×26×128
        ↓
    Flatten
    → Convert to 1D vector: 26×26×128 = 86,528
        ↓
    Fully Connected (1000 neurons)
    → Combine features for classification
        ↓
    Output (10 classes)
    → Softmax → Probabilities
    
    Real Example: Classifying Cat Photo
    
    Input: Cat photo
    ↓
    Conv 1: Detects edges (whiskers, ears)
    ↓
    Conv 2: Detects fur texture
    ↓
    Conv 3: Detects cat parts (triangular ears!)
    ↓
    FC: Combines evidence → "It's a cat!" (95%)
    """
    print("\n" + "=" * 70)
    print("3. CNN Architecture")
    print("=" * 70)
    print()
    print("💭 INTUITION: Assembly Line Manufacturing")
    print()
    print("   Raw Material (Input Image)")
    print("        ↓")
    print("   Station 1: Extract Features (Conv Layer)")
    print("   'Find edges, patterns'")
    print("        ↓")
    print("   Station 2: Filter (ReLU)")
    print("   'Keep good parts'")
    print("        ↓")
    print("   Station 3: Summarize (Pooling)")
    print("   'Reduce size, keep essentials'")
    print("        ↓")
    print("   Repeat (Deeper features)")
    print("        ↓")
    print("   Final Station: Decision (FC Layer)")
    print("   'Combine everything, classify'")
    print("        ↓")
    print("   Final Product (Classification)")
    print()
    
    print("🏗️ Example CNN Architecture (for ImageNet):")
    print()
    
    layers = [
        ("Input", "224×224×3", "RGB image"),
        ("Conv1", "224×224×32", "32 filters, 3×3, detect edges"),
        ("ReLU", "224×224×32", "Remove negatives"),
        ("MaxPool", "112×112×32", "Downsample by 2"),
        ("Conv2", "112×112×64", "64 filters, detect textures"),
        ("ReLU", "112×112×64", "Remove negatives"),
        ("MaxPool", "56×56×64", "Downsample by 2"),
        ("Conv3", "56×56×128", "128 filters, detect parts"),
        ("ReLU", "56×56×128", "Remove negatives"),
        ("MaxPool", "28×28×128", "Downsample by 2"),
        ("Flatten", "100,352", "Convert to 1D vector"),
        ("FC1", "1,000", "Fully connected layer"),
        ("FC2", "10", "Output classes"),
        ("Softmax", "10", "Class probabilities"),
    ]
    
    print("   Layer          Output Shape        Description")
    print("   " + "-" * 65)
    for name, shape, desc in layers:
        print(f"   {name:12s}   {shape:15s}   {desc}")
    print()
    
    print("📊 What Each Layer Does:")
    print()
    print("   1️⃣  Convolutional Layer:")
    print("      • Applies filters to detect patterns")
    print("      • Parameters: filter_size, num_filters, stride")
    print("      • Example: 32 filters of 3×3 = 32 different patterns")
    print()
    print("   2️⃣  ReLU Activation:")
    print("      • Removes negative values (max(0, x))")
    print("      • Adds non-linearity")
    print("      • Fast, works well in practice")
    print()
    print("   3️⃣  MaxPooling:")
    print("      • Takes maximum in each region (e.g., 2×2)")
    print("      • Reduces spatial size (downsampling)")
    print("      • Provides translation invariance")
    print()
    print("   Example: MaxPool 2×2")
    print("   Input (4×4):        Output (2×2):")
    print("   [1  2  | 3  4]      [6  8]")
    print("   [5  6  | 7  8]  →   [14 16]")
    print("   ------+------")
    print("   [9  10 | 11 12]")
    print("   [13 14 | 15 16]")
    print()
    print("   4️⃣  Fully Connected Layer:")
    print("      • Every neuron connects to every input")
    print("      • Combines all features for final decision")
    print("      • Usually at the end of network")
    print()
    
    print("🎯 Real Example: Cat Image Classification")
    print()
    print("   Input: 224×224 RGB cat photo")
    print()
    print("   After Conv1 (edge detection):")
    print("   • Detects whisker edges")
    print("   • Detects ear outlines")
    print("   • Detects fur boundaries")
    print()
    print("   After Conv2 (texture detection):")
    print("   • Combines edges → Fur texture")
    print("   • Combines edges → Striped patterns")
    print("   • Combines edges → Smooth surfaces")
    print()
    print("   After Conv3 (part detection):")
    print("   • Combines textures → Triangular ears!")
    print("   • Combines textures → Whisker clusters")
    print("   • Combines textures → Round eyes")
    print()
    print("   After FC Layer (decision):")
    print("   • Triangular ears? ✓")
    print("   • Whiskers? ✓")
    print("   • Fur texture? ✓")
    print("   → Prediction: 'Cat' (95% confidence)")
    print()
    
    print("💡 Design Principles:")
    print()
    print("   1. Start with small filters (3×3 or 5×5)")
    print("   2. Increase channels as you go deeper")
    print("      (3 → 32 → 64 → 128 → 256)")
    print("   3. Use MaxPool to reduce spatial size")
    print("   4. Stack multiple conv layers before pooling")
    print("   5. End with fully connected layers")
    print()
    
    print("🚀 Famous CNN Architectures:")
    print()
    print("   LeNet-5 (1998): Digit recognition, 5 layers")
    print("   AlexNet (2012): ImageNet winner, 8 layers, ReLU + dropout")
    print("   VGG-16 (2014): Very deep (16 layers), small 3×3 filters")
    print("   ResNet (2015): 50-152 layers, skip connections")
    print("   EfficientNet (2019): Optimized for efficiency")
    print()
    print("   Trend: Deeper networks + smart architectures = Better results")


# ============================================================================
# 4. Training Tips
# ============================================================================

def demo_training_tips():
    """
    Practical tips for training CNNs effectively.
    """
    print("\n" + "=" * 70)
    print("4. CNN Training Tips")
    print("=" * 70)
    print()
    
    print("💡 Data Augmentation (Critical for CNNs!):")
    print()
    print("   Problem: Need 1000s of labeled images (expensive!)")
    print()
    print("   Solution: Generate variations of existing images")
    print("   • Random rotation (±15°)")
    print("   • Random flip (horizontal)")
    print("   • Random crop")
    print("   • Random brightness/contrast")
    print("   • Color jittering")
    print()
    print("   Result: 1,000 images → 10,000+ variations!")
    print("   Impact: Reduces overfitting, improves generalization")
    print()
    
    print("🎯 Transfer Learning (The Secret Weapon):")
    print()
    print("   Instead of training from scratch:")
    print("   1. Start with pre-trained model (e.g., ResNet on ImageNet)")
    print("   2. Remove last layer")
    print("   3. Add your own classifier")
    print("   4. Fine-tune on your data")
    print()
    print("   Benefits:")
    print("   • Train with 100s of images instead of 1000s")
    print("   • Train in hours instead of days")
    print("   • Better accuracy (learns from ImageNet's 14M images)")
    print()
    print("   Real Example:")
    print("   Scratch: 70% accuracy, 2 days training")
    print("   Transfer learning: 95% accuracy, 2 hours training!")
    print()
    
    print("⚙️ Optimization Tips:")
    print()
    print("   1️⃣  Batch Size:")
    print("      • Small (16-32): Better generalization, slower")
    print("      • Large (128-256): Faster training, needs more memory")
    print("      • Start with 32, increase if GPU has memory")
    print()
    print("   2️⃣  Learning Rate:")
    print("      • Too high: Loss explodes")
    print("      • Too low: Training too slow")
    print("      • Sweet spot: 0.001 (Adam) or 0.01 (SGD)")
    print("      • Use learning rate scheduler (decay over time)")
    print()
    print("   3️⃣  Optimizer:")
    print("      • SGD: Simple, needs tuning")
    print("      • Adam: Adaptive, works out of the box")
    print("      • Start with Adam (lr=0.001)")
    print()
    print("   4️⃣  Regularization:")
    print("      • Dropout: Randomly drop 20-50% of neurons")
    print("      • Batch Normalization: Normalize layer inputs")
    print("      • Weight Decay: L2 penalty on weights")
    print()
    
    print("🐛 Debugging Checklist:")
    print()
    print("   Loss not decreasing:")
    print("   • Learning rate too high/low")
    print("   • Bad initialization")
    print("   • Gradient vanishing/exploding")
    print()
    print("   Training good, validation bad (overfitting):")
    print("   • Add dropout")
    print("   • Add data augmentation")
    print("   • Reduce model complexity")
    print("   • Get more training data")
    print()
    print("   Training slow:")
    print("   • Use GPU (100x speedup!)")
    print("   • Increase batch size")
    print("   • Use mixed precision training")
    print("   • Use smaller image size")
    print()
    
    print("📊 Monitoring Training:")
    print()
    print("   Track these metrics:")
    print("   • Training loss (should decrease)")
    print("   • Validation loss (should decrease)")
    print("   • Validation accuracy (should increase)")
    print("   • Learning rate (if using scheduler)")
    print()
    print("   Good signs:")
    print("   ✅ Both losses decreasing")
    print("   ✅ Small gap between train and val loss")
    print("   ✅ Accuracy improving")
    print()
    print("   Bad signs:")
    print("   ❌ Train loss much lower than val loss (overfitting)")
    print("   ❌ Loss increasing or NaN (instability)")
    print("   ❌ No improvement after many epochs (stuck)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🖼️ CNNs for Image Classification\n")
    print("Learn why CNNs dominate computer vision!")
    print()
    
    demo_why_cnns()
    demo_convolution()
    demo_cnn_architecture()
    demo_training_tips()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why CNNs?
   - Dramatically fewer parameters than fully connected
   - Translation invariant (finds patterns anywhere)
   - Learns hierarchical features (edges → shapes → objects)

2. Convolution Operation:
   - Small filter slides over image
   - Detects patterns locally
   - Same filter reused everywhere (efficient!)

3. CNN Architecture:
   - Conv layers: Extract features
   - Pooling: Reduce size, provide invariance
   - FC layers: Make final decision
   - Stack deeper for better features

4. Training Tips:
   - Data augmentation: Generate variations
   - Transfer learning: Start with pre-trained model
   - Adam optimizer: lr=0.001
   - Dropout + BatchNorm: Prevent overfitting

Real Impact:
- ImageNet: 75% → 95%+ accuracy (CNNs)
- Face recognition, self-driving cars, medical imaging
- Transfer learning: 100 images can be enough!

PyTorch Example:
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 54 * 54, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 54 * 54)
        x = self.fc(x)
        return x
```

Next: Transfer learning with pre-trained models!
""")


if __name__ == "__main__":
    main()
