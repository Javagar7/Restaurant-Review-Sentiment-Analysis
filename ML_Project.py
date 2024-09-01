import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

#Importing data set

dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t',quoting = 3)

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#Creating bad of words

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#Splitting

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#Train model

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

#Confusion matrix

cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)

print('Accuracy score is ',ac*100)
