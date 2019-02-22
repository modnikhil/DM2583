import numpy as np
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn.svm as svm
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# sources: https://github.com/iolucas/nlpython/blob/master/blog/sentiment-analysis-analysis/svm.ipynb
# https://medium.com/nlpython/sentiment-analysis-analysis-part-2-support-vector-machines-31f78baeee09

# File Name declaration
train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"

train_set = open(train_set_file)
test_set = open(test_set_file)

# function that parses through a set, and modifies a given list that represents the set's labels
def parse_files(file_set, labels_list):
    file_list = []
    count = 0
    for _ ,line in enumerate(file_set):
        # ignore the first line
        if count != 0:
            review = line[line.find(',')+1:line.rfind(',')]
            score = float(line[line.rfind(', ')+1:].rstrip("\n"))
            # Can fine-tune the trimming of punctutation later if accuracy is to low
            # These are just preliminary filters
            review = review.replace('<br />', " ")
            review = review.replace('.', " ")
            review = review.replace('-', " ")
            # Remove punctuation, lowercase entire string, split with the (" ") delimiter, remove empty strings
            # remove punctuation
            review = review.translate(review.maketrans('','',string.punctuation))

            review = review.lower()
            review = " ".join(filter(None, review.split()))

            file_list.append(review)
            if score <= 3:
                labels_list.append('neg ')
            else:
                labels_list.append('pos ')
        count += 1
    return [indiv_review for indiv_review in file_list]

train_labels = []
review_tokens = parse_files(train_set, train_labels)

vectorizer = CountVectorizer()
vectorizer.fit(review_tokens)

# split data into training and test set with train_test_split function
x_train, x_test, y_train, y_test = train_test_split(review_tokens, train_labels, test_size=100, random_state=1234, shuffle=False)

# for unbalanced dataset
sm = SMOTE(random_state=12, ratio = 1.0, kind='svm')



# create svm classifier and train it
lsvm = LinearSVC()
lsvm = make_pipeline(sm, lsvm)
x_train_res, y_train_res = sm.fit_sample(vectorizer.transform(x_train), y_train)

lsvm.fit(x_train_res, y_train_res)

# get accuracy/performance of classifier
score = lsvm.score(vectorizer.transform(x_test), y_test)

print("SVM Classifier score: the classifier performed on the test set with an accuracy of " + str(score * 100) + " %")


# score = lsvm.score(onehot_enc.transform(x_test), y_test)
#print("SVM Classifier score: the classifier performed on the test set with an accuracy of " + str(score * 100) + " %")
#

y_pred = lsvm.predict(vectorizer.transform(x_test))
# print(y_pred)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["neg", "pos"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()




