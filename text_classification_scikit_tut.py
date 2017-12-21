# Text Tutorial from Sci-Kit Learn
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#exercise-3-cli-text-classification-utility

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

twenty_train.target_names

twenty_train.data

len(twenty_train.data)

twenty_train.filenames

len(twenty_train.filenames)

print("\n".join(twenty_train.data[0].split("\n")[:3]))

print(twenty_train.target_names[twenty_train.target[0]])

twenty_train.target[:10]

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)


import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

from sklearn import metrics

print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)
