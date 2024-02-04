# Part 1: How to handle text input

## Converting text into numbers

All Machine Learning Models are mathematical functions and they can only take numerical input. For a model to be able to handle text data it is necessary that we pass the text through a tokenizer first and then vectorize each token to generate token embeddings.

### What is a tokenizer

Tokenizer is a function that will split the input text into different segments and each segment will be unique.

### Are there different ways of splitting the text

There are different ways in which the text can be split. We can split at character-level, word-level, and also subword-level. Subword-level is when we break existing words into smaller constituents (plugin -> plug + ##in).

### Which splitting technique is more meaningful

Splitting at character-level makes the text lose all meaning. Splitting at word-level is the most meaningful but there are a LOT of words and it would lead to a huge vocabulary. Subword splitting is the most optimal because it leads to small vocabulary while still retaining a lot of the meaning.

### Why is it important for tokens to be meaningful and finite

Machine Learning works by looking for patterns. If the tokens are meaningful and finite- it reduces the search space of the model and it becomes easier for them to look for important/broader patterns.

### How do we get the embeddings of each token

With BERT the token embeddings are learnt during pre-training (which will be covered later). But before BERT there were other more intuitive and simpler ways to generate embeddings for tokens.

The key idea behind GloVe is to capture global statistical information about word co-occurrences in a corpus to generate meaningful word embeddings. It did this by-
1. Taking a huge text corpus and generating a co-occurence matrix.
2. Designing an objective function which minimized the difference between the product of embeddings for words (that co-occur frequently) and their corresponding co-occurrence probabilities in the matrix.
3. GloVe employs matrix factorization techniques to factorize the co-occurrence matrix X into two lower-dimensional matrices W and C. Each row of matrix W represents the vector representation (embedding) of a word as the "target" word, and each row of matrix C represents the vector representation of a word as the "context" word.
4. Adjusting the word embeddings to minimize the objective function using stochastic gradient descent.
5. The learned embeddings are extracted from the rows of the matrix W or C after training.

### Can we directly use these embeddings

Token embeddings capture semantic relationships between Tokens in the training corpus. These embeddings can be used to measure similarity and perform various natural language processing tasks.

### What are the differences between latent features and token embeddings

Latent features and token embeddings are both representations of data used in machine learning, but they differ in their contexts, applications, and the way they capture information:

1. Definition:
  + Latent Features: These are abstract, underlying representations learned by a model during training. Latent features are not directly observable but are inferred from the data to capture relevant patterns.
  + Token Embeddings: These are numeric representations of words or subword units in natural language processing (NLP). Token embeddings aim to capture semantic and contextual information about individual tokens.
2. Context and Specificity:
  + Latent Features: Generalize patterns in the entire dataset, representing abstract features that contribute to the model's understanding of the input.
  + Token Embeddings: Capture specific information about individual tokens, including their semantic meaning and contextual usage within a given sequence or sentence.
3. Dynamic vs. Static:
  + Latent Features: Can be dynamic, changing with the learning process as the model updates its internal representations during training.
  + Token Embeddings: Can be dynamic (contextual embeddings like BERT) or static (fixed embeddings like Word2Vec). Contextual embeddings adapt to the surrounding context, while static embeddings provide fixed representations.

In summary, latent features are abstract representations learned by a model during training for various tasks, while token embeddings specifically refer to numeric representations of words or subword units in NLP.

### Are latent features meaningless to humans

The interpretability of latent features depends on the complexity of the model. In simpler models, the learned features may align more closely with human-understandable patterns. In highly complex models, such as deep neural networks, the features can become intricate and abstract.

While the raw latent features may be hard to interpret, visualization techniques can help provide insights. Dimensionality reduction methods, such as t-SNE or PCA, can be used to visualize the distribution of latent features in a lower-dimensional space, aiding in human understanding.

### What is K-Nearest Neighbors

You can use k-nearest neighbors (KNN) on numerical representations generated by algorithms like Word2Vec, GloVe, or any other method. KNN is a simple and effective algorithm for finding similar items based on their vector representations.

The KNN algorithm stores the entire training dataset in memory. Each data point in the training set has associated labels for classification tasks or target values for regression tasks.

When a new input query is received, the algorithm calculates the distance between the query point and every point in the training dataset. Common distance metrics include Euclidean distance, Manhattan distance, or cosine similarity.

### Are approaches like KNN applied on latent features or token embeddings

If the task is to find similar word-units then token embeddings can be used.

If the task is find similarity between more complex structures, like an entire text document, then latent features would be required for applying KNN.
