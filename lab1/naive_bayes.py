import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
import tkinter

# File Name declaration
train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"
evaluation_set_file = "data/lab_evaluation.txt"

train_set = open(train_set_file)
test_set = open(test_set_file)
evaluation_set = open(evaluation_set_file)

# Each dictionary maps a word to its frequency in the class
ratings = [dict(), dict()]
num_words = 0
score_freqs = [0] * len(ratings)
priors = [0.0] * len(ratings)

# Confusion matrix lists
class_true = []
class_pred = []

###### TRAINING SET #######
# For each review, remove the noise, and populate our naive bayes table
for _ ,line in enumerate(train_set):
    review = line[line.find(',')+1:line.rfind(',')]
    # 0 = positive, 1 = negative
    score = 0 if float(line[line.rfind(', ')+1:].rstrip("\n")) > 3.0 else 1
    
    # Remove noise, remove punctuation, lowercase, split sentence into words
    review = review.replace('<br />', " ").replace('.', " ").replace('-', " ")
    review = review.translate(review.maketrans('','',string.punctuation)).lower()
    review = filter(None, review.split(" "))

    # Populate Naive Bayes frequency table
    for word in review:
        # Null Check
        if word in ratings[score]:
            ratings[score][word] += 1
        else:
            ratings[score][word] = 1
        # Standardize # of keys in each class dictionary
        if word not in ratings[(score + 1) % 2]:
            ratings[(score + 1) % 2][word] = 0
        num_words += 1
        score_freqs[score] += 1

# Evaluate priors
for i in range(0, len(ratings)):
    priors[i] = float(score_freqs[i]) / float(num_words)


###### TEST_SET #####
total_reviews = 0
correct = 0
ratings_len = [len(ratings[score].keys()) for score in range(0, len(ratings))]

for _ ,line in enumerate(test_set):
    review = line[line.find(',')+1:line.rfind(',')]
    # 0 = positive, 1 = negative
    actual_score = 0 if float(line[line.rfind(', ')+1:].rstrip("\n")) > 3.0 else 1
        
    # Remove noise, remove punctuation, lowercase, split sentence into words
    review = review.replace('<br />', " ").replace('.', " ").replace('-', " ")
    review = review.translate(review.maketrans('','',string.punctuation)).lower()
    review = list(filter(None, review.split(" ")))
    
    # Evaluate Posteriors
    posteriors =  [1000000.0] * len(ratings) # Pad starting posteriors to avoid rounding to 0
    for score in range(0,len(ratings)): 
        for word in review:
            # Null Check
            if word not in ratings[score]:
                ratings[score][word] = 0
            # Evaluate likelihoods with Additive Smoothing
            likelihood = (float(ratings[score][word] + 1.0)) / (float(score_freqs[score] + len(ratings[score].keys()) + 1.0))
            posteriors[score] *=  (100* likelihood)
        posteriors[score] *= priors[score]
    
    # Take argmax of posteriors to classify review
    pred_score = float(np.argmax(posteriors))
    class_true.append("Positive" if actual_score == 0 else "Negative")
    class_pred.append("Positive" if pred_score == 0 else "Negative")

    # Accuracy evaluators
    total_reviews += 1
    if (pred_score == actual_score):
        correct += 1

print("Test Set Accuracy:", float(correct)/float(total_reviews))

###### EVALUATION SET ######
for review_idx ,review in enumerate(evaluation_set):
    # Remove punctuation, lowercase, split sentence into words
    review = review.translate(review.maketrans('','',string.punctuation)).lower()
    review = list(filter(None, review.split(" ")))
    
    # Evaluate Posteriors
    posteriors =  [1000000.0] * len(ratings) # Pad starting posteriors to avoid rounding to 0
    for score in range(0,len(ratings)): 
        for word in review:
            # Null Check
            if word not in ratings[score]:
                ratings[score][word] = 0
            # Evaluate likelihoods with Additive Smoothing
            likelihood = (float(ratings[score][word] + 1.0)) / (float(score_freqs[score] + len(ratings[score].keys()) + 1.0))
            posteriors[score] *=  (100* likelihood)
        posteriors[score] *= priors[score]
    
    # Take argmax of posteriors to classify review
    pred_class = "Positive" if float(np.argmax(posteriors)) == 0.0 else "Negative"
    print("Review #", review_idx, " Class: " + pred_class)

####### CONFUSION MATRIX EVALUATION ########
y_actu = pd.Series(class_true, name='Actual')
y_pred = pd.Series(class_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(df_confusion)

def plot_confusion_matrix(df_confusion, title='Confusion Matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_conf_norm)
plt.show()