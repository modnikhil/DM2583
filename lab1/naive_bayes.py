import numpy as np
import string

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
    pred_score = float(np.argmax(posteriors))
    #print("Line Num: ", review_idx, " Score: ", pred_score)

