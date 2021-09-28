import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump
dataset = pd.read_csv('data.csv', encoding='ISO-8859-1');

dataset.head()

x_target = dataset.iloc[:, 1]
y_data = dataset.iloc[:, 0]

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])

text_clf = text_clf.fit(x_target, y_data)

dump(text_clf, 'model.joblib')
