# Part 2: How do neural networks learn

## What even are neural networks

Neural networks are a class of machine learning models. These models consist of interconnected nodes, also known as neurons or artificial neurons, organized into layers. Neural networks are capable of learning complex patterns and representations from data, making them particularly effective in tasks such as pattern recognition, classification, regression, and more.

### Why are they called "neural" networks

Neural networks are called "neural" because they draw inspiration from the structure and functioning of biological neural networks in the human brain. However, the resemblance is very minute.

### What is a neuron

Neurons are the basic building blocks of neural networks. They receive input signals, perform a computation, and produce an output.

### Is one neuron enough

A single neuron, on its own, is quite limited in terms of its ability to perform complex tasks. However, it can perform simple tasks that involve linear transformations and basic non-linearities.

### How many is enough

The number of neurons needed for complex tasks in a neural network depends on several factors, including the complexity of the task, the nature of the data, and the architecture of the network. There is no one-size-fits-all answer, as different tasks may require different network architectures and sizes.

### What are activation functions and why are they inserted between linear functions

Neurons in a neural network often apply an activation function to the weighted sum of their inputs. Common activation functions include sigmoid, hyperbolic tangent (tanh), and rectified linear unit (ReLU).

Activation functions play a crucial role in artificial neural networks by introducing non-linearity to the output of a neuron. These functions determine whether a neuron should be activated (fire) or not based on the weighted sum of its inputs and a bias term. The non-linearity introduced by activation functions allows neural networks to model complex relationships and make them capable of learning from data.

### What is a softmax function

The softmax function is a mathematical function that converts a vector of numerical values into a probability distribution. It is commonly used in the output layer of a neural network for multiclass classification tasks (more on this in Part 4).

## How to create a good model

Creating a good machine learning model involves a systematic approach that encompasses understanding the problem, preprocessing the data, selecting appropriate algorithms, training and fine-tuning the model, and evaluating its performance.

Here, we will mainly focus on training and fine-tuning.

### How do we intialize all the parameters in the model

The initialization of parameters in a machine learning model, especially in neural networks, is a crucial aspect of training. Proper initialization helps the model converge faster and avoid issues like vanishing or exploding gradients.

1. Zero Initialization: Set all weights and biases to zero. While this is a simple approach, it's generally not recommended for deep networks as it can lead to symmetry problems (all neurons in a layer learning the same features).
2. Random Initialization: Initialize weights with small random values. This helps break the symmetry and allows each neuron to learn different features. Commonly used in practice.
3. Xavier/Glorot Initialization: A popular initialization method, especially for sigmoid or hyperbolic tangent (tanh) activation functions. It sets the weights using a Gaussian distribution.
4. He Initialization: Similar to Xavier, but adapted for ReLU activation functions.

BERT uses Xavier/Glorot Initialization.

### How to train a model

Backpropagation is a supervised learning algorithm used to train neural networks. It involves adjusting the weights and biases of the network based on the error between the predicted output and the actual target values. This process is repeated iteratively to minimize the error.

### How does the model update the weights on its own

### How much monitoring is required

### Is more training always good

### What is data drift

## Why do neural networks work

### What is the universal approximation theorem

### What is Information Theory of Deep Learning

### What is Noam Chomsky's Cognitive Learning theory
