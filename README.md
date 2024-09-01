Restaurant Review Sentiment Analysis
Overview
This project is designed to analyze restaurant reviews and classify them as either positive or negative based on the text content. Using a dataset of restaurant reviews, the project preprocesses the text, converts it into numerical features using a Bag of Words model, and then applies a Naive Bayes classifier to predict the sentiment of the reviews.

Features
Text Preprocessing:
Removal of non-alphabetic characters.
Conversion of text to lowercase.
Tokenization and stemming of words.
Stopword removal with the exception of the word "not" to maintain negative sentiment.
Bag of Words Model:
Creation of a feature matrix using the CountVectorizer with a maximum of 1500 features.
Sentiment Classification:
Implementation of a Gaussian Naive Bayes classifier to predict whether a review is positive or negative.
Model Evaluation:
Use of confusion matrix and accuracy score to evaluate the model's performance.
Technologies Used
Python: Programming language for implementation.
Pandas: Data manipulation and analysis.
NLTK: Natural Language Toolkit for text preprocessing.
Scikit-learn: Machine learning library for model building and evaluation.
Dataset
Restaurant_Reviews.tsv: A tab-separated file containing restaurant reviews along with labels indicating whether the review is positive (1) or negative (0).


Future Enhancements
Explore additional machine learning models like Support Vector Machines (SVM) or Random Forest.
Implement a deep learning approach using LSTM or CNN for better accuracy.
Integrate the model into a web application for real-time sentiment analysis.
