#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"
sample_set_file = "data/lab_sample.txt"


train_set = open(train_set_file)
test_set = open(test_set_file)
sample_set = open(sample_set_file)

data = []
data_labels = []
with train_set as f:
    for line in f:
        review = line[line.find(',')+1:line.rfind(',')]
        score = float(line[line.rfind(', ')+1:].rstrip("\n"))
        data.append(review) 
        data_labels.append(str(score))

vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() 

#Load dataset
wine = datasets.load_wine()

# print the names of the 13 features
print("Features: ", wine.data)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target)

data = list(np.asarray(data))
data_labels = list(np.asarray(data_labels))

#X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3333,random_state=109) # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=0.3333,random_state=109) # 70% training and 30% test


print(type(wine.data))
print(type(wine.target))
print(type(data))
print(type(data_labels))

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))