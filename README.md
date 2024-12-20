# Incrementally Training CNN Models on MNIST Data

This repository has 3 files to build and optimize Convolutional Neural Networks (CNNs) trained on the MNIST dataset. We progressively develop three CNN models, each trying to improve the test accuracy so that it stays above 99.4% for at least the last 3 epochs. We want to finalize the model with under 8k parameters and achieve the target in 15 epochs or lesser. 

The first iteration is to start with a basic CNN implementation to ensure correct model skeleton, we then advance to explore other techniques to tweak the model further. 

## Notebooks

### 1. Model Iteration - 1 : Basic Model
**Notebook**: `MNIST_CNN_Model_Iteration_1.ipynb`

**Features**:
- Basic CNN architecture with under 8k parameters - focus on establishing basic skeleton
- Simple MaxPooling, ReLU activations, Batch Normalization and dropouts

### 2. Model Iteration - 2: Advanced Model with Image augmentation
**Notebook**: `MNIST_CNN_Model_Iteration_2.ipynb`

**Features**:
- Added Random Rotation to the images. 
- Adding other augmentation techniques (horizontal Flip) did not improve accuracy
- Varying dropouts also do not improve accuracy. 

### 3. Final Model Iteration
**Notebook**: `MNIST_CNN_Model_Iteration_3.ipynb`

**Features**:
- Advanced data augmentation:
  - Random Rotation (-7° to 7°)
- Learning rate scheduling with warm cosine scheduling 
- Consistently achieves >99.4% test accuracy

**Training Details**:
- 15 epochs
- Batch size: 32
- Initial learning rate: 0.015
- Momentum: 0.9

## Results

### Basic Model (Model 1)
- Parameters: 7,934
- Best Training Accuracy: 99.06%
- Best Test Accuracy: 99.41%
- Analysis: Basic skeleton with under 8k parameters, underfitted model that can be trained further

### Advanced Model (Model 2)
- Parameters: 7,934
- Best Training Accuracy: 98.71%
- Best Test Accuracy: 99.35%
- Analysis: Improved generalization with random rotation added, but target accuracy not achieved. 

### Final Model (Model 3)
- Parameters: 7,934
- Best Training Accuracy: 98.97%
- Best Test Accuracy: 99.50%

- Consistently maintains >99.4% test accuracy 
- Meets all target criteria:
  - Test accuracy > 99.4%
  - Training completed in ≤15 epochs
  - Parameters < 8000

## Requirements
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm