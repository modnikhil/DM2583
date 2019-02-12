import numpy as np
import string
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

# File Name declaration
train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"

train_set = open(train_set_file)
test_set = open(test_set_file)

"""
Things to keep track of:
    For Training:
        - Total num of all words in reviews:    int                         [total_word_count]
        - Score Frequencies:                    dict{float/int, int}        [score_freqs]
        - Score Prior Probabilities:            dict{float/int, float}      [score_probs]
        - Word Score Frequencies:               list of 5 dict{string, int} [word_scorings(?)]
  For Testing:
        Posterior = (Prior * Likelihood) / Evidence
        - We choose the class (score in our case) that maximizes our Posterior
        - We ignore Evidence in our calculation since it is the same across all the Posteriors we are comparing to each other
        - Prior is P(1.0), P(2.0), etc.
            - Calculation for P(1.0) = score_freqs[0]/total_word_count
        - Likelihood is P(word_1, word_2,...,word_i|1.0), P(word_1, word_2,...,word_i|2.0), etc.
            - Calculation for P(word_1, word_2,...,word_i|1.0) = (word_scorings[0][word_1]/score_freqs[0]) * (word_scorings[0][word_2]/score_freqs[0]) *...* (word_scorings[0][word_i]/score_freqs[0])
"""

# each dictionary maps a word to the # of time
# Enumerate through every line of the file and split into bag of words
# TODO(?): Modularize this to accept any file

def word_feats(words):
    return dict([(word, True)  for word in words])

positive_vocab = []
negative_vocab = []
iterations = 0
for _ ,line in enumerate(train_set):
    review = line[line.find(',')+1:line.rfind(',')]
    score = float(line[line.rfind(', ')+1:].rstrip("\n"))
    # Can fine-tune the trimming of punctutation later if accuracy is to low
    # These are just preliminary filters
    review = review.replace('<br />', " ")
    review = review.replace('.', " ")
    review = review.replace('-', " ")
    # Remove punctuation, lowercase entire string, split with the (" ") delimiter, remove empty strings
    #remove punctuation
    review = review.translate(review.maketrans('','',string.punctuation))
    #lowercase
    review = review.lower()

    # print("review: " + review)
    review = filter(None, review.split(" "))
    # print(review)
    _rating = score
    # print(_rating)
    for word in review:
        # print("word: " + word)
        #words = filter(None, string.lower(review.translate(string.punctuation)).split(" "))
        if(_rating <= 3):
            negative_vocab.append(word)
        else:
            positive_vocab.append(word)
    # print("n_list: " + str(negative_vocab))
    # print("p_list: " + str(positive_vocab))
    # iterations += 1
    # if(iterations == 25):
    #     break


# returns a list of tuples
def featurize(vocab, label):
    features = []
    for word in vocab:
        curr_dict  = {}
        curr_dict[word]  = label
        features.append( (curr_dict, True) )
    return features



# print("positive_vocab: " + str(l))
positive_features = featurize(positive_vocab, 'pos')
# positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
# print(positive_vocab)
negative_features = featurize(negative_vocab, 'neg')
# print(positive_features)
# print(negative_features)
# print("neg_fe: "  + str(negative_features))
# print("pos_fe: " + str(positive_features))


train_set = list(negative_features) + list(positive_features)

classifier = NaiveBayesClassifier.train(train_set)
tot_pos = 0
tot_neg = 0

def features(words):
    return dict([(word, True) for word in words])

# Clean up test_set
# for _ ,line in enumerate(test_set):
#     review = line[line.find(',')+1:line.rfind(',')]
#     score = float(line[line.rfind(', ')+1:].rstrip("\n"))
#     # Can fine-tune the trimming of punctutation later if accuracy is to low
#     # These are just preliminary filters
#     review = review.replace('<br />', " ")
#     review = review.replace('.', " ")
#     review = review.replace('-', " ")
#     # Remove punctuation, lowercase entire string, split with the (" ") delimiter, remove empty strings
#     #remove punctuation
#     review = review.translate(review.maketrans('','',string.punctuation))
#     #lowercase
#     review = review.lower()
#     review_text = review
#     # review = filter(None, review.split(" "))
#     review = review.split(' ')
#     # words = sentence.split(' ')
#     neg = 0
#     pos = 0
#     # print(classResult)
#     for word in review:
#         classResult = classifier.classify( word_feats(word))
#         # print(word + "  result: " + classResult)
#         if classResult == 'neg':
#             neg = neg + 1
#         if classResult == 'pos':
#             pos = pos + 1
#     if(pos > neg):
#         tot_pos += 1
#     else:
#         tot_neg += 1
#     tot_pos += pos
#     tot_neg += neg
#     pos = 0
#     neg = 0
# print('Positive: ' + str(float(tot_pos)))
# print('Negative: ' + str(float(tot_neg)))


# print("TOTAL POSITIVE (ABOVE 2.5): " + str(tot_pos))
# print("TOTAL NEGATIVE (BELOW 2.5): " + str(tot_neg))


# Predict
neg = 0
pos = 0
sentence = "Awful"
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    print(classResult)
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1

        tot_pos += 1
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))


