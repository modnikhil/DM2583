import numpy as np
import string

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


# Enumerate through every line of the file and split into bag of words
# TODO(?): Modularize this to accept any file
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
    filter(None, review.split(" "))
    print(review)
    #words = filter(None, string.lower(review.translate(string.punctuation)).split(" "))
