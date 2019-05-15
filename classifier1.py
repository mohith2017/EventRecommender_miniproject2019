import pandas as pd
df = pd.read_csv('FINAL.csv', encoding="latin9")
df.head()
print("done")


df = df[pd.notnull(df['Event_name'])]
df.info()


print("done")

col = ['Tag', 'Event_name']
df = df[col]
df.columns
df.columns = ['Tag', 'Event_name']

df['category_id'] = df['Tag'].factorize()[0]
from io import StringIO
category_id_df = df[['Tag', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Tag']].values)

print("done")

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('Tag').Event_name.count().plot.bar(ylim=0)
# plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Event_name).toarray()
labels = df.category_id
features.shape

#################################################

from sklearn.feature_selection import chi2
import numpy as np

N = 2
for Tag, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Tag))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

#####################################################

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['Event_name'], df['Tag'], test_size=0.2)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)


#####################################################

print(clf.predict(count_vect.transform(["ios workshop"])))

if clf.predict(count_vect.transform(["ios workshop"]))=="Coding":
    print("1")
