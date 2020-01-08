import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
print vocabulary

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


vec=MeanEmbeddingVectorizer(word2vec)
Matris=vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Matris, y, test_size=0.4, random_state=2)

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
