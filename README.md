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

# Importing libraries
from numpy import array
	from numpy import argmax
	import nltk
	nltk.download('punkt')
	import pandas as pd
	from nltk.tokenize import TweetTokenizer, sent_tokenize
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.model_selection import train_test_split
	from sklearn.decomposition import LatentDirichletAllocation
	from google.colab import files
	import io
	import numpy as np
	

	from google.colab import drive
	drive.mount('/content/drive')
	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn import metrics 
	from sklearn.ensemble import RandomForestClassifier
  
  # Train and test data (here, we have previously classified data into train and test)
  df_train_sen= pd.read_csv("/content/drive/My Drive/Thesis/105 reports/train_sen.csv",encoding='ISO-8859-1')df_test_sen= pd.read_csv("/content/drive/My Drive/Thesis/105 reports/test_sen_complete.csv",encoding='ISO-8859-1')
  
  # Vectorizing the text into numerical vectors
  #TFIDF vectorizervectorizer = TfidfVectorizer(stop_words="english")X = vectorizer.fit_transform(df_train_sen["sentences"].tolist())lda = LatentDirichletAllocation(n_components=10, random_state=0)X_lda = lda.fit_transform(X)Y_target = df_train_sen['Sentence topic']
  
  # LDA for dimensional reduction
  X_train = X_lday_train = Y_targetX_test = vectorizer.fit_transform(df_test_sen["sentences"].tolist())lda = LatentDirichletAllocation(n_components=10, random_state=0)X_test_lda = lda.fit_transform(X_test)Y_test = df_test_sen['sentence_topic']
  
  # Classification and Prediction Using Logistic Regression
  # Create Logistic Regression
	model = OneVsRestClassifier(SVC(C=0.0001, kernel="rbf", gamma="auto",probability=True))
	clf = SVC(kernel='rbf', probability=True) 
	clf.fit(X_train, y_train)
    y_hat_prob_lr = clf.predict_proba(X_test_lda)
    yhat_lr = clf.predict(X_test_lda)
    
   # Classification and Prediction Using Decision Tree
   # Create Decision Tree classifer object
	clf_dt = DecisionTreeClassifier()
    clf_dt = clf_dt.fit(X_train,y_train)
	yhat_dt = clf_dt.predict(X_test_lda)

  # Predict the response for test dataset
	y_hat_prob_dt = clf_dt.predict_proba(X_test_lda)
  
  # Classification and Prediction Using Random Forest Classifier
  # Create the model with 100 trees
	modelrf = RandomForestClassifier(n_estimators=100, 
	                               bootstrap = True,
	                               max_features = 'sqrt')
  # Fit on training data
	modelrf.fit(X_train,y_train)
	yhatRF = modelrf.predict(X_test_lda)
	y_hat_prob_rf = modelrf.predict_proba(X_test_lda)
	
