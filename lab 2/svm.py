import numpy as np
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

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

            file_list.append(review)
            if score <= 3:
                labels_list.append('neg ')
            else:
                labels_list.append('pos ')
        count += 1
    return file_list

train_labels = []
test_labels = []
review_list = parse_files(train_set, train_labels)
test_list = parse_files(test_set, test_labels)

# -- for testing, prints the reviews and whether their respective class/label --
# curr = 0
# for val in review_list:
#     print(val)
#     print(train_labels[curr])
#     curr += 1


# load the module to transform review inputs into binary vectors using MulriLabelBinarizer class
onehot_enc = MultiLabelBinarizer()
onehot_enc.fit(review_list)

# split data into training and test set with train_test_split function
x_train, x_test, y_train, y_test = train_test_split(review_list, train_labels, test_size=100, random_state=None)

# create svm classifier and train it
lsvm = LinearSVC()
lsvm.fit(onehot_enc.transform(x_train), y_train)

# get accuracy/performance of classifier
score = lsvm.score(onehot_enc.transform(x_test), y_test)

print("SVM Classifier score: the classifier performed on the test set with an accuracy of " + str(score * 100) + " %")