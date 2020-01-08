import re
from sklearn.datasets import load_files
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC


documents=[]
comments_data = load_files(r"docs\tel\xiaomiredmi5plus")    ##write the name of the folder as the first item from the folders in docs/tel
X_train, y_train = comments_data.data, comments_data.target
for sen in range(0, len(X_train)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X_train[sen]))

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

comments_data = load_files(r"docs\tel\iphone6s")        ##write the name of the folder as the second item from the folders in docs/tel
X_test, y_test = comments_data.data, comments_data.target

documents2=[]
for sen in range(0, len(X_test)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X_test[sen]))

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
    documents2.append(document)

vectorizer = CountVectorizer(ngram_range=(3,10),min_df=50, analyzer = 'char')
X_train = vectorizer.fit_transform(documents).toarray()
vocab=vectorizer.get_feature_names()
vectorizer = CountVectorizer(ngram_range=(3,10),min_df=50, analyzer = 'char', vocabulary=vocab)
X_test=vectorizer.fit_transform(documents2).toarray()
tfidfconverter = TfidfTransformer()
X_traintfidf = tfidfconverter.fit_transform(X_train).toarray()
X_testtfidf=tfidfconverter.fit_transform(X_test).toarray()
X_train=X_traintfidf
X_test = X_testtfidf
print ("n-gram with LinearSVC 1.urun eğitim 2. urun test\n")
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))




vectorizer = CountVectorizer(ngram_range=(3,10),min_df=50, analyzer = 'char')
X_train = vectorizer.fit_transform(documents2).toarray()
vocab=vectorizer.get_feature_names()
vectorizer = CountVectorizer(ngram_range=(3,10),min_df=50, analyzer = 'char', vocabulary=vocab)
X_test=vectorizer.fit_transform(documents).toarray()
tfidfconverter = TfidfTransformer()
X_traintfidf = tfidfconverter.fit_transform(X_train).toarray()
X_testtfidf=tfidfconverter.fit_transform(X_test).toarray()
X_train=X_traintfidf
X_test = X_testtfidf
print ("n-gram with LinearSVC 2.urun eğitim 1. urun test\n")
svm = LinearSVC()
svm.fit(X_train, y_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_train,y_pred))
print(classification_report(y_train,y_pred))
print(accuracy_score(y_train, y_pred))

