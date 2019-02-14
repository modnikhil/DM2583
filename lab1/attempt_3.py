import numpy as np
import string
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
from nltk.corpus import stopwords


# File Name declaration
train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"

# nltk.download('stopwords')

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
    #return dict([(word, True)  for word in words])
    return {words:True}


one_star_vocab = []
two_star_vocab= []
three_star_vocab = []
four_star_vocab = []
five_star_vocab = []

iterations = 0
stop_words = set(stopwords.words('english'))

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
    # print(review)

    for word in review:
        if(_rating >= 1 and _rating <= 1.99):
            one_star_vocab.append(word)
            one_star_vocab.append(word)
            one_star_vocab.append(word)
            one_star_vocab.append(word)
            one_star_vocab.append(word)
            one_star_vocab.append(word)
            one_star_vocab.append(word)
        elif(_rating >= 2 and _rating <= 2.99):
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
            two_star_vocab.append(word)
        elif(_rating >= 3  and _rating <= 3.99):
            # add all 3 stars
            three_star_vocab.append(word)
            three_star_vocab.append(word)
            three_star_vocab.append(word)
            three_star_vocab.append(word)
            three_star_vocab.append(word)
            three_star_vocab.append(word)
            three_star_vocab.append(word)
        elif(_rating >= 4 and _rating <= 4.99):
            # add all 4 stars
            four_star_vocab.append(word)
            four_star_vocab.append(word)
        else:
            # add all 5.0's
            five_star_vocab.append(word)

    # print("n_list: " + str(negative_vocab))
    # print("p_list: " + str(positive_vocab))
    # iterations += 1
    # if(iterations == 25):
    #     break


# returns a list of tuples
# def featurize(vocab, label):
#     features = []
#     for word in vocab:
#         curr_dict  = {}
#         curr_dict[word]  = label
#         features.append( (curr_dict, True) )
#     return features



# print("positive_vocab: " + str(l))
one_star_features = [(word_feats(rating), 'one') for rating in one_star_vocab]
two_star_features = [(word_feats(rating), 'two') for rating in two_star_vocab]
three_star_features = [(word_feats(rating), 'three') for rating in three_star_vocab]
four_star_features = [(word_feats(rating), 'four') for rating in four_star_vocab]
five_star_features = [(word_feats(rating), 'five') for rating in five_star_vocab]

# print(str(negative_features)

# positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
# print(positive_vocab)
# negative_features = featurize(negative_vocab, 'neg')
# print(positive_features)
# print(negative_features)
# print("neg_fe: "  + str(negative_features))
# print("pos_fe: " + str(positive_features))


train_set = one_star_features + two_star_features + three_star_features + four_star_features + five_star_features

classifier = NaiveBayesClassifier.train(train_set)
tot_pos = 0
tot_neg = 0
correct = 0
total = 0
def features(words):
    return dict([(word, True) for word in words])

# Clean up test_set
for _ ,line in enumerate(test_set):
    review = line[line.find(',')+1:line.rfind(',')]
    actual_score = float(line[line.rfind(', ')+1:].rstrip("\n"))
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
    review_text = review
    review = list(filter(None, review.split(" ")))

    for word in review:
        if(word in stop_words):
            review.remove(word)

    # Predict
    one_star = 0
    two_star = 0
    three_star = 0
    four_star = 0
    five_star = 0
    # print(classResult)


    for word in review:
        classResult = classifier.classify( word_feats(word))
        # print(word + "  result: " + classResult)
        if classResult == 'one':
            one_star += 1
        if classResult == 'two':
            two_star += 1
        if classResult == 'three':
            three_star += 1
        if classResult == 'four':
            four_star += 1
        if classResult  == 'five':
            five_star += 1

    one_star = float(one_star)/len(review)
    two_star = float(two_star)/len(review)
    three_star = float(three_star)/len(review)
    four_star = float(four_star)/len(review)
    five_star = float(five_star)/len(review)
    negative = False
    positive  = False
    neg_sum = one_star  + two_star + three_star
    pos_sum = four_star + five_star
    if(three_star > neg_sum and three_star > pos_sum):
        if( neg_sum > pos_sum ):
            negative = True
        else:
            positive = True
    elif(pos_sum > neg_sum):
        positive = True
    elif(neg_sum > pos_sum):
        negative = True

    if(actual_score <= 4 and negative):
        correct += 1
    elif(actual_score > 4 and positive):
        correct += 1
    total += 1
    positive = False
    negative = False

    # print("***************REVIEW************************")
    # print(review)
    # print('One star: ' + str(float(one_star)/len(review)))
    # print('Two star: ' + str(float(two_star)/len(review)))
    # print('Three star: ' + str(float(three_star)/len(review)))
    # print('Four star: ' + str(float(four_star)/len(review)))
    # print('Five star: ' + str(float(five_star)/len(review)))
    one_star = 0
    two_star = 0
    three_star = 0
    four_star = 0
    five_star = 0




print("******************ACCURACY******************")
print(correct / total)

# print(len(stop_words))
# print(stop_words)


# print("TOTAL POSITIVE (ABOVE 2.5): " + str(tot_pos))
# print("TOTAL NEGATIVE (BELOW 2.5): " + str(tot_neg))


# Predict
# one_star = 0
# two_star = 0
# three_star = 0
# four_star = 0
# five_star = 0
# review = "waste"
# review = review.replace('<br />', " ")
# review = review.replace('.', " ")
# review = review.replace('-', " ")
# # Remove punctuation, lowercase entire string, split with the (" ") delimiter, remove empty strings
# #remove punctuation
# review = review.translate(review.maketrans('','',string.punctuation))
# #lowercase
# review = review.lower()
# review_text = review
# review = list(filter(None, review.split(" ")))
# # print("Review: " + sentence)
# # print("words: " + str(words))
# for word in review:
#     classResult = classifier.classify(word_feats(word))
#     # print(classifier.prob_classify(word_feats(word) ))
#     # print(classResult)
#     if classResult == 'one':
#         one_star += 1
#     if classResult == 'two':
#         two_star += 1
#     if classResult == 'three':
#         three_star += 1
#     if classResult == 'four':
#         four_star += 1
#     if classResult  == 'five':
#         five_star += 1

# print(review)
# print('One star: ' + str(float(one_star)/len(review)))
# print('Two star: ' + str(float(two_star)/len(review)))
# print('Three star: ' + str(float(three_star)/len(review)))
# print('Four star: ' + str(float(four_star)/len(review)))
# print('Five star: ' + str(float(five_star)/len(review)))



