import joblib
from joblib import load
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
model = load("model.joblib")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(["testing testing fake"])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#category_label = model.predict(X_train_tfidf)[0]
category_label = model.predict(["testing testing fakae msg"])[0]

print(category_label)
