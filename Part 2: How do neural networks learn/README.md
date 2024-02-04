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

The process of updating weights in a machine learning model occurs during the training phase, and it is typically guided by an optimization algorithm. One common optimization algorithm used for training neural networks is Stochastic Gradient Descent (SGD) or its variants. The primary steps in weight updating involve computing gradients, adjusting weights based on these gradients, and iterating through the process.

### Why do we need optimisation algorithms

The search space for weights in a neural network is extremely high-dimensional, and exploring all possible combinations would be computationally infeasible. Instead of brute force, iterative optimization algorithms are employed to efficiently search the parameter space, adjusting weights based on feedback from the training data.

### How much monitoring is required

1. Initial Setup and Configuration: During the initial stages, human involvement is necessary to set up the training process, configure hyperparameters, and define the architecture of the model. This includes choosing an appropriate optimization algorithm, initializing weights, and selecting relevant evaluation metrics.
2. Data Preprocessing and Exploration: Humans play a crucial role in preprocessing and exploring the dataset. This involves handling missing values, scaling or normalizing features, encoding categorical variables, and identifying potential issues such as class imbalances or outliers.
3. Model Training: While the model is training, monitoring is essential to ensure that the training process is progressing as expected. This may involve checking for signs of convergence, observing changes in the loss function, and analyzing intermediate results.
4. Hyperparameter Tuning: Human intervention is often required for hyperparameter tuning. This includes adjusting learning rates, regularization parameters, and other hyperparameters based on the performance observed during training and validation.
5. Error Analysis: During or after training, humans analyze the model's errors on the validation or test set. This analysis helps identify patterns of misclassifications, understand where the model struggles, and potentially guide further adjustments or feature engineering.
6. Model Interpretability: Understanding the interpretability of the model's predictions may involve human inspection, especially for models where interpretability is crucial. Techniques such as feature importance analysis or visualization can assist in interpreting the model's decisions.
7. Monitoring for Anomalies or Issues: Human monitoring is important for detecting anomalies or unexpected behavior during training. This could include issues like sudden spikes or drops in performance, which may require investigation and adjustments.
8. Early Stopping: Early stopping is a technique where training is halted if the model performance on a validation set stops improving. Deciding when to stop training may involve human judgment and is often based on a trade-off between training time and model performance.
9. Deployment Considerations: When preparing to deploy the model, humans may need to monitor its behavior on real-world data to ensure it generalizes well and meets the desired performance criteria.

### Is more training always good

No, more training does not always guarantee improved performance in machine learning models. The relationship between training duration and model performance is influenced by several factors, and there are situations where additional training may not provide significant benefits or could even lead to overfitting. Here are some key considerations:

1. Convergence: If a model has already converged to a satisfactory performance level on the training and validation sets, additional training may not lead to substantial improvements. Convergence occurs when the model has learned the underlying patterns in the data, and further training may only fine-tune the weights without significant gains.
2. Overfitting: Prolonged training can lead to overfitting, especially if the model becomes too complex or if the training data includes noise. Overfitting occurs when the model performs well on the training data but fails to generalize to new, unseen data. Regularization techniques and early stopping are often employed to prevent overfitting.
3. Diminishing Returns: In some cases, the performance gains achieved by additional training diminish over time. The model may reach a point where further updates to the weights have minimal impact on the overall performance.
4. Computational Resources: Training a model for an extended period may require substantial computational resources, and the benefits of additional training must be weighed against the associated costs. In practical scenarios, there may be constraints on computational resources or time.
5. Validation Performance: Monitoring the model's performance on a validation set is crucial. If the validation performance plateaus or starts to degrade, it may indicate that additional training is not beneficial. Early stopping based on validation performance is a common practice to avoid overtraining.
6. Learning Rate and Optimization Algorithm: The learning rate and choice of optimization algorithm can impact the model's training dynamics. Adjusting these hyperparameters may be necessary to achieve optimal convergence without overfitting.
7. Task Complexity: The complexity of the machine learning task plays a role in determining the required training duration. More complex tasks may require longer training, but there is no universal rule, and the optimal training duration depends on the specific characteristics of the problem.

In summary, the decision to continue training a model should be guided by careful monitoring of performance on both the training and validation sets. It's important to strike a balance between achieving convergence and avoiding overfitting.

### What is data drift

Data drift refers to the phenomenon where the statistical properties of the input data to a machine learning model change over time, causing a degradation in the model's performance. This change in data distribution can lead to a mismatch between the training data and the data the model encounters during deployment, affecting the model's ability to generalize to new, unseen examples.

### How to handle data drift

Online learning is a machine learning paradigm that can be particularly helpful in addressing data drift. Online learning, also known as incremental or streaming learning, is a training approach where the model is updated continuously as new data becomes available. This makes it well-suited for scenarios where the data distribution is non-stationary, and the underlying patterns change over time.

## Why do neural networks work

Neural networks consist of non-linear activation functions applied to weighted sums of input features. This non-linearity allows them to learn and represent highly complex and non-linear relationships within the data, making them powerful function approximators.

Neural networks are organized into layers, allowing them to learn hierarchical representations of features. Lower layers capture basic features, while higher layers learn more abstract and complex representations. This hierarchical feature learning enables neural networks to automatically discover relevant features from raw data.

### What is the universal approximation theorem

The Universal Approximation Theorem states that a neural network with a single hidden layer and a sufficient number of neurons can approximate any continuous function to arbitrary accuracy. This theorem highlights the expressive power of neural networks.

### Do neural networks understand things the way we do

No, they don't. Noam Chomsky's Cognitive Learning theory posits that much of language learning is innate, hard-wired in the brain, and less reliant on external input. In contrast, neural networks in machine learning are trained using large datasets, relying heavily on the exposure to examples and patterns within the data.

### Is deep learning like curve fitting

Deep learning can be seen as a form of curve fitting, but it is a highly sophisticated and complex form of it. The basic idea of curve fitting is to find a function that best represents a given set of data points. In deep learning, particularly in the context of neural networks, the model learns a complex mapping from input data to output predictions, which can be thought of as fitting a function to the training data.

While deep learning shares similarities with curve fitting, it goes beyond simple curve fitting methods. Deep learning models can automatically learn hierarchical features and representations from data, adapt to diverse tasks, and handle large-scale and high-dimensional data.

The term "curve fitting" is often associated with simpler regression techniques, while deep learning encompasses a broader set of methods that can handle more complex tasks, such as image recognition, natural language processing, and reinforcement learning, among others.

### Do we understand how neural networks process information

Deep Neural Networks are black boxes. The term "black box" refers to a system or model whose internal workings are not transparent or easily understandable. A black box model is characterized by the difficulty in interpreting how it makes predictions or decisions based on input data. The lack of transparency can make it challenging for humans to comprehend the rationale behind specific outcomes generated by the model.

Efforts are ongoing in the field of explainable artificial intelligence (XAI) to develop techniques and methods that enhance the interpretability of black box models. Explainability tools, feature importance analyses, and model-agnostic interpretability methods are some approaches aimed at making the decision-making process of black box models more understandable to users, stakeholders, and regulators.

### What is Information Theory of Deep Learning

The "information theory of deep learning" generally refers to the application of concepts and principles from information theory to understand and analyze aspects of deep learning models, training processes, and representations. Information theory, a branch of applied mathematics and electrical engineering, deals with quantifying and analyzing information and communication systems. In the context of deep learning, researchers often leverage information theory to gain insights into how neural networks function and learn. Here are some key aspects of the information theory of deep learning:

1. Entropy and Uncertainty: Entropy is a fundamental concept in information theory, representing the amount of uncertainty or disorder in a set of data. In deep learning, entropy can be used to measure the uncertainty associated with predictions made by a model. Higher entropy indicates greater uncertainty, and lower entropy implies more confident predictions.
2. Cross-Entropy Loss: Cross-entropy is a common loss function used in classification tasks. It is derived from information theory and measures the dissimilarity between predicted probability distributions and true distributions (one-hot encoded labels). Minimizing cross-entropy effectively encourages the model to assign high probabilities to the correct class.
3. Mutual Information: Mutual information quantifies the degree of dependency between two random variables. In deep learning, mutual information can be used to assess the relationship between input features and the target variable, providing insights into how well the model captures relevant information from the input data.
4. Information Bottleneck: The information bottleneck principle in deep learning suggests that the intermediate representations learned by a neural network should strike a balance between preserving relevant information from the input data and compressing it to focus on essential features for the task. This concept is related to the trade-off between underfitting and overfitting.
5. Compression and Generalization: From an information-theoretic perspective, effective learning involves the compression of information. A good model should be able to compress input data into a more concise representation that retains essential information for making predictions. This ties into the model's generalization ability to perform well on new, unseen data.
6. Capacity and Expressiveness: Information theory can be used to analyze the capacity and expressiveness of neural networks. The capacity of a model refers to its ability to represent a wide range of functions. Analyzing the information content in the weights and activations helps understand how well the model captures patterns and complexity in the data.
7. Quantifying Redundancy and Irrelevance: Information theory provides tools for quantifying redundancy and irrelevance in the input data or model representations. Identifying and eliminating redundant information can contribute to more efficient learning and better generalization.
8. Regularization and Noise Tolerance: Information theory principles guide the design of regularization techniques that encourage models to be more robust to noise and variations in the input data. Techniques like dropout, which introduces noise during training, are inspired by principles related to information and uncertainty.

References-
1. [Stanford Seminar - Information Theory of Deep Learning, Naftali Tishby](https://www.youtube.com/watch?v=XL07WEc2TRI&t=1888s)
