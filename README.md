# Sentence-Level-LDA-Topic-Modelling
Example tutorial on using Latent Diritchlet Allocation (LDA) algorithm for topic classification for text data

Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of a topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

There has been a lot of talks and tutorial where they use LDA for topic modeling at the document level. However, there could be a situation where we would be interested to classify one sentence into one topic. The concept is not much different in this case as well. For the document level topic classification, we used one document as one entity, and here we consider one sentence as one entity. Here, we will cover this topic.

LDA is based on the underlying assumptions: the distributional hypothesis, (i.e. similar topics make use of similar words) and the statistical mixture hypothesis (i.e. documents talk about several topics) for which a statistical distribution can be determined. The purpose of LDA is mapping each document in our corpus to a set of topics that covers a good deal of the words in the document.

Here, the goal of this tutorial is to classify the sentences in medical reports into specific clinical topics. This is a part of medical reports where the sentences are classified into two topics.

1. Tumor Characteristics
2. Nontumor characteristics

The data are annotated in terms of sentences as 1 for "tumor characteristics" and 0 for "non-tumor characteristics".

For the vectorization of the text, TFIDF vectors have been used. Vectorization is a process of converting text into numerical vectors so that the mathematical operations required for the machine learning models could be carried out. There are several methods of doing such a vectorization process. TFIDF is one of the popular methods due to its success and not so complex concepts underlying it.

The TFIDF vectors thus produced are sparse in nature and require some compression for better results. When passed through LDA, it also serves as dimensionality reduction.

After we get complex LDA vectors, we then classify them into topics using machine learning classification algorithms. For this purpose, we have used three classification algorithms:

1. Logistic Regression
2. Random Forest
3. Decision Trees

