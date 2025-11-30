# Module 7: Deep Learning with PyTorch

**Goal**: Build and train neural networks for real-world AI applications.

**Why PyTorch**: Industry standard, Pythonic, great for research and production.

## 📚 What You'll Learn

### Neural Network Fundamentals
- Feedforward networks (multilayer perceptrons)
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (MSE, Cross-Entropy)
- Backpropagation (automatic differentiation)
- Optimizers (SGD, Adam, AdamW)
- Regularization (Dropout, Batch Norm)

### PyTorch Essentials
- Tensors (like NumPy but with GPU support)
- Autograd (automatic gradients)
- nn.Module (building blocks)
- DataLoader (efficient data loading)
- Training loops
- Model saving/loading

### Advanced Architectures
- Convolutional Neural Networks (CNNs) - Images
- Recurrent Neural Networks (RNNs) - Sequences
- Transformers - NLP, Vision
- Autoencoders - Compression, denoising
- GANs - Generative models

### Computer Vision
- Image classification
- Object detection
- Transfer learning (pretrained models)
- Data augmentation

### Natural Language Processing
- Word embeddings (Word2Vec, GloVe)
- Text classification
- Sequence models (LSTM, GRU)
- Attention mechanisms
- Transformers (BERT-style)

## 🎯 Real-World Projects

- **Image Classifier**: Cats vs Dogs, CIFAR-10
- **Sentiment Analysis**: Movie reviews, tweets
- **Object Detection**: YOLO, Faster R-CNN
- **Text Generation**: Character/word-level RNN
- **Recommendation System**: Collaborative filtering
- **Anomaly Detection**: Credit card fraud

## 📂 Module Structure

```
07-deep-learning/
├── README.md (you are here)
├── fundamentals/
│   ├── 01_pytorch_basics.py         # Tensors, autograd
│   ├── 02_neural_network_scratch.py # Build NN from scratch
│   ├── 03_pytorch_nn_module.py      # Using nn.Module
│   ├── 04_training_loop.py          # Standard training pattern
│   └── 05_optimization.py           # Optimizers, schedulers
├── computer_vision/
│   ├── 01_image_classification.py   # CNN basics
│   ├── 02_transfer_learning.py      # Use pretrained models
│   ├── 03_data_augmentation.py      # Improve generalization
│   └── 04_object_detection.py       # YOLO/Faster R-CNN
├── nlp/
│   ├── 01_word_embeddings.py        # Word2Vec, embeddings
│   ├── 02_text_classification.py    # Sentiment analysis
│   ├── 03_sequence_models.py        # RNN, LSTM, GRU
│   └── 04_attention_transformers.py # Attention mechanism
├── advanced/
│   ├── 01_autoencoders.py           # Compression, denoising
│   ├── 02_variational_ae.py         # VAE for generation
│   ├── 03_gans.py                   # Generative models
│   └── 04_reinforcement_learning.py # DQN basics
└── projects/
    ├── image_classifier/            # CIFAR-10 classifier
    ├── sentiment_analyzer/          # IMDB reviews
    └── recommendation_system/       # MovieLens
```

## 💡 PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Learning Curve | Easier (Pythonic) | Steeper |
| Debugging | Easier | Harder |
| Research | Preferred | Also used |
| Production | Excellent | Excellent |
| Community | Growing fast | Larger |

**Choice**: PyTorch for this course (easier to learn, research-friendly)

## 🔧 GPU Support

**Why GPU?**
- 10-100x faster than CPU for deep learning
- Essential for large models

**Setup**:
```bash
# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"

# Install with CUDA support (if you have NVIDIA GPU)
poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Neural Network Architecture Patterns

**Image Classification**:
```
Input (Image) → Conv Layers → Pooling → FC Layers → Output (Classes)
```

**Text Classification**:
```
Input (Text) → Embedding → RNN/Transformer → FC Layers → Output (Classes)
```

**Sequence-to-Sequence**:
```
Input Sequence → Encoder → Context Vector → Decoder → Output Sequence
```

## 🎓 Learning Path

1. **PyTorch Basics** → Tensors, autograd
2. **Build from Scratch** → Understand backprop
3. **Use nn.Module** → Build efficiently
4. **Computer Vision** → CNNs, transfer learning
5. **NLP** → RNNs, transformers
6. **Advanced** → GANs, VAEs
7. **Projects** → Portfolio pieces

## ⚡ Best Practices

**1. Data Pipeline**
```python
# Use DataLoader for efficient batching
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**2. Model Architecture**
```python
# Inherit from nn.Module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Define forward pass
        return x
```

**3. Training Loop**
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**4. Save/Load**
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

## 🚀 Quick Start

```bash
# Install PyTorch
poetry add torch torchvision torchaudio

# Start with basics
poetry run python 07-deep-learning/fundamentals/01_pytorch_basics.py

# Build your first neural network
poetry run python 07-deep-learning/fundamentals/03_pytorch_nn_module.py

# Image classification project
poetry run python 07-deep-learning/projects/image_classifier/train.py
```

## 🎯 Expected Outcomes

After this module:
- ✅ Understand neural network internals
- ✅ Build models with PyTorch
- ✅ Train on GPU efficiently
- ✅ Use transfer learning
- ✅ Build CNNs for computer vision
- ✅ Build RNNs/Transformers for NLP
- ✅ Deploy models in production

---

**Ready to build AI?** Let's start with PyTorch! 🚀
