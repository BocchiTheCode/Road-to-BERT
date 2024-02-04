# Part 4: BERT

## What is BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a pre-trained natural language processing (NLP) model introduced by Google researchers Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova in 2018.

Here are key features of BERT:
  + Transformer Architecture: BERT is based on the transformer architecture, which allows it to efficiently capture contextual information and dependencies in both directions (left-to-right and right-to-left) in a sequence of words.
  + Pre-training on Large Corpus: BERT is pre-trained on a massive amount of text data. During pre-training, the model learns to predict missing words in a sentence (masked language modeling) and understand the relationships between words.
  + Contextualized Embeddings: BERT produces contextualized word embeddings, meaning that the representation of a word can vary based on its context within a sentence. This allows BERT to understand the nuances of word meanings in different contexts.
  + Fine-tuning for Downstream Tasks: After pre-training, BERT can be fine-tuned for specific downstream tasks such as text classification, named entity recognition, question answering, and more. Fine-tuning adapts the pre-trained model to perform well on specific applications with smaller, task-specific datasets.
  + State-of-the-Art Performance: BERT achieved state-of-the-art performance on a wide range of NLP benchmarks, surpassing previous models in tasks like question answering and sentiment analysis.
  + Open Source Implementation: BERT is open-source, and pre-trained models are available for researchers and developers to use. The availability of pre-trained models has led to the development of various BERT-based applications and solutions.

## How was BERT trained

BERT (Bidirectional Encoder Representations from Transformers) was trained using a two-step process: pre-training and fine-tuning. The pre-training phase involves training the model on a large corpus of text data using unsupervised learning, and the fine-tuning phase adapts the pre-trained model to specific downstream tasks using supervised learning on smaller, task-specific datasets.

1. Pre-training:
  + Corpus Selection: BERT was pre-trained on a massive amount of text data collected from the internet. This corpus included a diverse range of content to ensure that the model learned general language patterns.
  + Masked Language Modeling (MLM): BERT uses a masked language modeling objective during pre-training. In this process, random words in the input sentences are masked, and the model is trained to predict the masked words based on the context provided by the surrounding words. This bidirectional training allows the model to understand contextual dependencies.
  + Next Sentence Prediction (NSP): BERT also incorporates a next sentence prediction task. Pairs of sentences are provided to the model during training, and the model learns to predict whether the second sentence follows the first in the original document or not. This helps the model capture relationships between sentences.

2. Fine-tuning:
  + Text Classification: Classifying text into predefined categories (e.g., sentiment analysis, spam detection).
  + Named Entity Recognition (NER): Identifying and classifying entities (e.g., person names, locations) in text.
  + Question Answering: Extracting answers from passages given questions.
  + Sentence Similarity: Determining the similarity between pairs of sentences.

Fine-tuning allows BERT to leverage its pre-trained knowledge and adapt to the nuances of specific tasks with smaller amounts of labeled data.

### How was Masked Language Modelling done with BERT

In the MLM process, some of the words in this sentence are randomly selected and masked. The model's objective is to predict or fill in these masked words based on the context provided by the surrounding words.

  + Original Sentence: "The quick brown fox jumps over the lazy dog."
  + Masked Sentence: "The quick brown [MASK] jumps over the [MASK] dog."

Now, the model needs to predict the masked words. The actual words in these positions are "fox" and "lazy." During training, the model learns to predict these masked words by considering the context of the surrounding words.

### How was Next Sentence Prediction done with BERT

1. During pre-training, BERT takes sentence pairs as input. These pairs consist of two consecutive sentences from the same document, separated by a [SEP] token. Example: "The cat sat on the mat. [SEP] It was a sunny day."
2. The sentence pair representation is then processed through multiple layers of transformer encoders. These encoders capture contextual information and bidirectional dependencies within the sentences.
3. The representation of the special [CLS] token (at the beginning of the first sentence) is used as a pooled representation of the entire sentence pair.
4. This pooled representation is then fed into a binary classification layer to predict whether the second sentence in the pair is the actual next sentence or a randomly chosen one from the dataset.

The NSP objective helps the model capture relationships between sentences and understand the overall context of a document; but there are studies which have shown that the NSP objective might not significantly improve performance on many downstream tasks compared to other pre-training objectives, such as masked language modeling (MLM).

### Why is Masked Language Modelling useful

1. MLM trains the model to predict masked words based on the surrounding context. As a result, the model learns contextualized representations for words, capturing how the meaning of a word can change in different contexts. This is crucial for tasks that require understanding nuanced language use.
2. Polysemy refers to words having multiple meanings. MLM helps the model handle polysemy by training it to predict the correct meaning of a word based on the context in which it appears. This improves the model's ability to distinguish between different senses of a word.
3. By predicting masked words, the model learns semantic relationships between words. It understands the semantics of words not just based on their individual meanings but also in the context of a sentence. This contributes to a more profound understanding of language semantics.
4. MLM allows for unsupervised pre-training on a large corpus of unlabeled text. This is valuable as it leverages the vast amount of freely available text on the internet, enabling the model to learn general language patterns without the need for task-specific labeled data.

## What are the downstream tasks that BERT can perform

BERT has been extensively used for-
1. Text Classification: Sentiment analysis, Spam detection, Topic classification
2. Named Entity Recognition (NER): Identifying entities in text.
3. Question Answering: Extracting answers from passages given a question.
4. Text Similarity and Semantic Matching: Determining similarity between sentences or documents. Paraphrase identification.
5. Machine Translation: Leveraging BERT for context-aware translation models.
6. Summarization: Generating abstractive summaries of documents.
7. Language Understanding: Extracting information from unstructured text. Understanding user queries in chatbots and virtual assistants.
8. Conversational AI: Enhancing chatbots and conversational agents with improved understanding of context.
9. Information Retrieval: Improving search engine results by understanding user queries and document relevance.

Here, we will only focus on text classification.

### What is multi-class classification

Multiclass classification is a type of machine learning classification task where the goal is to categorize instances into three or more classes or categories. In contrast to binary classification, which involves distinguishing between two classes (e.g., spam or not spam), multiclass classification involves assigning instances to one of multiple possible classes. Key characteristics of multiclass classification:

1. There are more than two mutually exclusive classes or categories that an instance can belong to.
2. Each instance is assigned to one and only one class. In other words, an instance cannot belong to multiple classes simultaneously.
3. Common evaluation metrics for multiclass classification include accuracy, precision, recall, F1 score, and confusion matrices.

### What is cross-entropy loss

Cross-entropy loss, also known as log loss, is a commonly used loss function in machine learning, particularly for classification problems. It measures the performance of a classification model whose output is a probability distribution over classes. Cross-entropy loss increases as the predicted probability diverges from the actual label.

The cross-entropy loss is derived from the concept of information theory, specifically the Kullback-Leibler (KL) divergence, which measures how one probability distribution differs from another. The formulation of cross-entropy loss involves calculating the expected negative log-likelihood of the true distribution P under the predicted distribution Q.

## What are the different variants of BERT

BERT has inspired the development of various variants and extensions, each designed to address specific challenges or improve performance in certain domains.

### What is DistilBERT

DistilBERT is a smaller and faster version of BERT created by distillation techniques. It is designed to retain much of the performance of BERT while being more computationally efficient and suitable for deployment in resource-constrained environments.

### What is RoBERTa

RoBERTa is an extension of BERT that introduces changes to the pre-training tasks, training data, and hyperparameters. It omits the Next Sentence Prediction (NSP) task, uses dynamic masking during pre-training, and incorporates larger mini-batches. These modifications lead to improved performance on various downstream tasks.

### What is crammedBERT

An English-language model pretrained like BERT, but with less compute (only trained for 24 hours on a single A6000 GPU). This work sheds light on how to train a BERT-like model from scratch with limited compute.

References-
1. [Intuitively Understanding the Cross Entropy Loss](https://youtu.be/Pwgpl9mKars?si=daFFNoGm8W252ldx)
2. [crammedBERT](https://github.com/JonasGeiping/cramming)
