from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from  sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import nltk
import re

comments_data = load_files(r"docs\tel\10k")
X, y = comments_data.data, comments_data.target

documents=[]
for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()
    document = ' '.join(document)
    document = document.lower()
    documents.append(document)

stop   = stopwords.words('turkish')

for sen in documents:
    sen = [x for x in sen if x not in stop]


vectorizerbow = CountVectorizer(min_df=30)
bow=vectorizerbow.fit_transform(documents).toarray()

vectorizerngram = CountVectorizer(ngram_range=(3,10),min_df=50, analyzer = 'char')
ngram = vectorizerngram.fit_transform(documents).toarray()

tfidfconverter = TfidfTransformer()
ngramtfidf = tfidfconverter.fit_transform(ngram).toarray()
bowtfidf = tfidfconverter.fit_transform(bow).toarray()


ngram_X_train, ngram_X_test, ngram_y_train, ngram_y_test = train_test_split(ngramtfidf, y, test_size=0.4, random_state=2)

bow_X_train, bow_X_test, bow_y_train, bow_y_test = train_test_split(bowtfidf, y, test_size=0.4, random_state=2)

print ("n-gram with Random Forest Classifier\n")
RFclassifier = RandomForestClassifier(n_estimators=250, random_state=0)
RFclassifier.fit(ngram_X_train, ngram_y_train)
ngram_y_pred = RFclassifier.predict(ngram_X_test)
print(confusion_matrix(ngram_y_test,ngram_y_pred))
print(classification_report(ngram_y_test,ngram_y_pred))
print(accuracy_score(ngram_y_test, ngram_y_pred))
print ("-------------------------------------------------------------\n")

print ("n-gram with MultinomialNB\n")
MNBclassifier = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
MNBclassifier.fit(ngram_X_train, ngram_y_train)
MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
ngram_y_pred = MNBclassifier.predict(ngram_X_test)
print(confusion_matrix(ngram_y_test,ngram_y_pred))
print(classification_report(ngram_y_test,ngram_y_pred))
print(accuracy_score(ngram_y_test, ngram_y_pred))
print ("-------------------------------------------------------------\n")

print ("n-gram with LinearSVC\n")
svm=LinearSVC()
svm.fit(ngram_X_train, ngram_y_train)
ngram_y_pred = svm.predict(ngram_X_test)
print(confusion_matrix(ngram_y_test,ngram_y_pred))
print(classification_report(ngram_y_test,ngram_y_pred))
print(accuracy_score(ngram_y_test, ngram_y_pred))
print ("-------------------------------------------------------------\n")
print ("n-gram with LogisticRegression\n")
logreg=LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(ngram_X_train, ngram_y_train)
ngram_y_pred = logreg.predict(ngram_X_test)
print(confusion_matrix(ngram_y_test,ngram_y_pred))
print(classification_report(ngram_y_test,ngram_y_pred))
print(accuracy_score(ngram_y_test, ngram_y_pred))
print ("-------------------------------------------------------------\n")


print ("n-gram with SGDClassifier\n")
SGD=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
SGD.fit(ngram_X_train, ngram_y_train)
ngram_y_pred = SGD.predict(ngram_X_test)
print(confusion_matrix(ngram_y_test,ngram_y_pred))
print(classification_report(ngram_y_test,ngram_y_pred))
print(accuracy_score(ngram_y_test, ngram_y_pred))
print ("-------------------------------------------------------------\n")



print ("Bag-of-words with Random Forest Classifier\n")
RFclassifier = RandomForestClassifier(n_estimators=250, random_state=0)
RFclassifier.fit(bow_X_train, bow_y_train)
bow_y_pred = RFclassifier.predict(bow_X_test)
print(confusion_matrix(bow_y_test,bow_y_pred))
print(classification_report(bow_y_test,bow_y_pred))
print(accuracy_score(bow_y_test, bow_y_pred))
print ("-------------------------------------------------------------\n")

print ("Bag-of-words with MultinomialNB\n")
MNBclassifier = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
MNBclassifier.fit(bow_X_train, bow_y_train)
MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
bow_y_pred = MNBclassifier.predict(bow_X_test)
print(confusion_matrix(bow_y_test,bow_y_pred))
print(classification_report(bow_y_test,bow_y_pred))
print(accuracy_score(bow_y_test, bow_y_pred))
print ("-------------------------------------------------------------\n")

print ("Bag-of-words with LinearSVC\n")
svm=LinearSVC()
svm.fit(bow_X_train, bow_y_train)
bow_y_pred = svm.predict(bow_X_test)
print(confusion_matrix(bow_y_test,bow_y_pred))
print(classification_report(bow_y_test,bow_y_pred))
print(accuracy_score(bow_y_test, bow_y_pred))
print ("-------------------------------------------------------------\n")

print ("Bag-of-words with LogisticRegression\n")
logreg=LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(bow_X_train, bow_y_train)
bow_y_pred = logreg.predict(bow_X_test)
print(confusion_matrix(bow_y_test,bow_y_pred))
print(classification_report(bow_y_test,bow_y_pred))
print(accuracy_score(bow_y_test, bow_y_pred))
print ("-------------------------------------------------------------\n")


print ("Bag-of-words with SGDClassifier\n")
SGD=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
SGD.fit(bow_X_train, bow_y_train)
bow_y_pred = SGD.predict(bow_X_test)
print(confusion_matrix(bow_y_test,bow_y_pred))
print(classification_report(bow_y_test,bow_y_pred))
print(accuracy_score(bow_y_test, bow_y_pred))
print ("-------------------------------------------------------------\n")



df = pd.read_csv(r'docs\10k.csv')
df = df[pd.notnull(df['tag'])]
my_tags = ['neg','pos']


X = df.post
y = df.tag

all_words = [nltk.word_tokenize(sent) for sent in X]

for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('turkish')]




word2vec = Word2Vec(all_words, min_count=10)

vocabulary = word2vec.wv.vocab


class MyTokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)

        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


a=MeanEmbeddingVectorizer(word2vec)
b=a.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(b, y, test_size=0.4, random_state=2)

print ("Word2Vec with Random Forest Classifier\n\n")
RFclassifier = RandomForestClassifier(n_estimators=250, random_state=0)
RFclassifier.fit(X_train, y_train)
y_pred = RFclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print ("-------------------------------------------------------------\n")



print ("Word2Vec with LinearSVC\n\n")
svm=LinearSVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print ("-------------------------------------------------------------\n")
print ("Word2Vec with LogisticRegression\n\n")
logreg=LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print ("-------------------------------------------------------------\n")


print ("Word2Vec with SGDClassifier\n\n")
SGD=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
SGD.fit(X_train, y_train)
y_pred = SGD.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print ("-------------------------------------------------------------\n")
