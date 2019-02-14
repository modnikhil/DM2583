import numpy as np
import string
import copy
import nltk
from nltk.corpus import stopwords

# File Name declaration
train_set_file = "data/lab_train.txt"
test_set_file = "data/lab_test.txt"
sample_set_file = "data/lab_sample.txt"
evaluation_set_file = "data/lab_evaluation.txt"


train_set = open(train_set_file)
test_set = open(test_set_file)
sample_set = open(sample_set_file)
evaluation_set = open(evaluation_set_file)

nltk.download('stopwords')

# each dictionary maps a word to the # of time
positive = dict()
negative = dict()
ratings = [positive, negative]
num_words = 0
score_freqs = [0,0]
score_probs = [0.0, 0.0]

stop_words = set(stopwords.words('english'))

for _ ,line in enumerate(train_set):
    review = line[line.find(',')+1:line.rfind(',')]
    score = 0 if float(line[line.rfind(', ')+1:].rstrip("\n")) > 3.0 else 1
        
    review = review.replace('<br />', " ")
    review = review.replace('.', " ")
    review = review.replace('-', " ")
    review = review.translate(review.maketrans('','',string.punctuation))
    #lowercase
    review = review.lower()
    review = filter(None, review.split(" "))

    for word in review:
        if word in ratings[score]:
            ratings[score][word] += 1
        else:
            ratings[score][word] = 1
        if word not in ratings[(score + 1) % 2]:
            ratings[(score + 1) % 2][word] = 0
        num_words += 1
        score_freqs[score] += 1

for num in range(0, 2):
    score_probs[num] = float(score_freqs[num]) / float(num_words)

print("number of words", num_words)
print(score_freqs)
print(score_probs)

total_reviews = 0
correct = 0
ratings_len = [len(ratings[score].keys()) for score in range(0,2)]
# now clean up our test_set -----
for _ ,line in enumerate(test_set):
    review = line[line.find(',')+1:line.rfind(',')]
    actual_score = 0 if float(line[line.rfind(', ')+1:].rstrip("\n")) > 3.0 else 1
        
    review = review.replace('<br />', " ")
    review = review.replace('.', " ")
    review = review.replace('-', " ")
    
    review = review.translate(review.maketrans('','',string.punctuation))
    #lowercase
    review = review.lower()
    review = list(filter(None, review.split(" ")))
    
    
    posteriors =  [1000000.0] *2
    for score in range(0,2): 
        for word in review:
            if word not in ratings[score]:
                ratings[score][word] = 0
            likelihood = float(ratings[score][word] + 1.0) 
            likelihood /= float(score_freqs[score] + len(ratings[score].keys()) + 1.0)
            posteriors[score] *=  (100* likelihood)
        posteriors[score] *= score_probs[score]
    
    pred_score = float(np.argmax(posteriors))
    if (pred_score == actual_score) :
        correct += 1
    total_reviews += 1

print(float(correct)/float(total_reviews))


total_reviews = 0
correct = 0

for review_num ,line in enumerate(evaluation_set):
    review = line
    review = review.translate(review.maketrans('','',string.punctuation))
    #lowercase
    review = review.lower()
    review = list(filter(None, review.split(" ")))
    
    posteriors =  [1000000.0] *2
    for score in range(0,2): 
        for word in review:
            if word not in ratings[score]:
                ratings[score][word] = 0
            likelihood = float(ratings[score][word] + 1.0) 
            likelihood /= float(score_freqs[score] + len(ratings[score].keys()) + 1.0)
            posteriors[score] *= (100 * likelihood)
            
               
        posteriors[score] *= score_probs[score]
    
    
    pred_score = float(np.argmax(posteriors))
    print("Line Num: ", review_num, " Score: ", pred_score)

