import pandas as pd
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time
start = time. time()


#Read csv
df = pd.read_csv('FINAL.csv', encoding="utf-8")
df.head()
print("done")

############################Get credentials for firebase
cred = credentials.Certificate('D:/DESKTOP/MSRIT/Third year/6th sem/Mini project/ML algos/recommender-miniproject-firebase-adminsdk-lpz5s-96b76f1c6d.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://recommender-miniproject.firebaseio.com/'
})
ref = db.reference('recommender-miniproject')
############################Sort into categories and plot graph based on dataset

#Remove null values
df = df[pd.notnull(df['Event_name'])]
df.info()
print("done")


#Use only necessary columns, i.e., 'Tag' and 'Event_name'
col = ['Tag', 'Event_name']
df = df[col]
df.columns
df.columns = ['Tag', 'Event_name']


#Convert tags to Integer IDs
df['category_id'] = df['Tag'].factorize()[0]

#Sort the values according to tags
from io import StringIO
category_id_df = df[['Tag', 'category_id']].drop_duplicates().sort_values('category_id')
print(category_id_df)

#Store these tags in a dict
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Tag']].values)
print(id_to_category)
print("done")

#Plot the Graph of the density of each Data type
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Tag').Event_name.count().plot.bar(ylim=0)
plt.show()

#Sort the 'Event_name's according to their term frequency and take log of it(tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Event_name).toarray()
labels = df.category_id
features.shape

#######################################Generate uni and bigrams

from sklearn.feature_selection import chi2
import numpy as np

#To check correlatio between two terms
N = 2
for Tag, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  #Define unigrams and bigrams
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

  print("# '{}':".format(Tag))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

#############################################Initial wrong prediction

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['Event_name'], df['Tag'],train_size=0.7)

#Convert into matrix
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)

clf = MultinomialNB().fit(X_train_tfidf, y_train)



print(clf.predict(count_vect.transform([ref.child("Sentence").get()])))



#####################################################Compare the ML models' accuracies
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df)

###################################################Plot the models' accuracy graph
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()
###################################################Predict on test and train

from sklearn.model_selection import train_test_split

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

####################################################Confusion matrix based on LinearSVC

from sklearn.metrics import confusion_matrix


conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Tag.values, yticklabels=category_id_df.Tag.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

###################################################Correct the wrongly predicted values and print it
from IPython.display import display

for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 6:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Tag', 'Event_name']])
      print('')

##################################################

model.fit(features, labels)

#################################################Correctly generated uni and bigrams
from sklearn.feature_selection import chi2

N = 2
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(Product))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

  ################################################Corrected prediction with updation on firebase


  texts = [ref.child("Sentence").get()]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)

switcher = {
         "Cultural":0,
         "Sports":1,
         "Coding":2,
        "Business":3,
    }



for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print(switcher[id_to_category[predicted]])
  ref.update({'value':switcher[id_to_category[predicted]]})

##########################################



  from sklearn import metrics
print(metrics.classification_report(y_test, y_pred,
                                    target_names=df['Tag'].unique()))
end = time. time()
print(str(end - start)+" seconds to complete!")








# if clf.predict(count_vect.transform(["DJ"]))=="Coding":
#     print("1")
