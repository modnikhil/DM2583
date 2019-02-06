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

# each dictionary maps a word to the # of time
one_stars = dict()
two_stars = dict()
three_stars = dict()
four_stars = dict()
five_stars = dict()
ratings = [one_stars, two_stars, three_stars, four_stars, five_stars]
num_words = 0
score_freqs = [0, 0, 0, 0, 0]
score_probs = [0.0, 0.0, 0.0, 0.0, 0.0]
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
    review = filter(None, review.split(" "))
    # print(review)
    _rating_index = int(score - 1)

    for word in review:
        if word in ratings[_rating_index]:
            ratings[_rating_index][word] += 1
        else:
            ratings[_rating_index][word] = 1
        num_words += 1
        score_freqs[_rating_index] += 1

        #words = filter(None, string.lower(review.translate(string.punctuation)).split(" "))

for num in range(0, 5):
    score_probs[num] = float(score_freqs[num]) / float(num_words)

print(ratings)
print("number of words", num_words)
print(score_freqs)
print(score_probs)

# now clean up our test_set -----
for _ ,line in enumerate(test_set):
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
    review = filter(None, review.split(" "))
    # print(review)
    _rating_index = int(score - 1)

result = 0.0
posteriors =  [0.0, 0.0, 0.0, 0.0, 0.0]
for rating in range(1, 6):
    for ithword in review:
        # frequency = total number of times a word has been rated a score (1-5)
        # divide frequency by the number of times THAT rating has been given to any word
        # amount of times this word has been rated 1.000 - 5.00 divided by the total amount it's been  rated in all 

likelyhood for 1/ score frequencies 


