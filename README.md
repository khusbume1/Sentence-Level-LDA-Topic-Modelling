# Sentence-Level-LDA-Topic-Modelling
Example tutorial on using Latent Diritchlet Allocation (LDA) algorithm for topic classification for text data

Here, the goal of this tutorial is to classify the sentences in medical reports into specific clinical topics. This is a part of medical reports where the sentences are classified into two topics.
1. Tumor Characteristics
2. Non tumor characteristics

The data are annotated in terms of sentences as 1 for "tumor characteristics" and 0 for "non tumor characteristics".

For the vectorization of the text, TFIDF vectors have been used.

The vectorized text has been passed through Latent Diritchlet Allocation (LDA) for the task of topic classification. 

Unlike other LDA tutorials, the unit of this work is sentence instead of documents. Instead of classifying documents into topics, here we classify sentences into topics.

Since, this is a binary classification, we have used three machine learning models and one of them is logistic regression. The other two models are "Random Forest" and "Decision Trees"
