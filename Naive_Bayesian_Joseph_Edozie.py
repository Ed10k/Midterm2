# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:06:46 2017

@author: xfang13
Edited by Joseph Edozie
"""
from os import walk
import numpy as np

seed = 12345
number_of_training_emails = 500

ham_dir = 'Ham'
All_Ham = []
for (dirpath, dirnames, filenames) in walk(ham_dir):
    if filenames != []:
        for ham in filenames:
            with open(dirpath + '//' + ham) as f:  # changed the backwards slashes to forward slashes since I am on Mac
                # read the file as a big string
                All_Ham.append(f.read())

spam_dir = 'Spam'
All_Spam = []
for (dirpath, dirnames, filenames) in walk(spam_dir):
    if filenames != []:
        for spam in filenames:
            with open(dirpath + '//' + spam,
                      encoding='latin-1') as f:  # changed the backwards slashes to forward slashes since I am on Mac
                # read the file as a big string
                All_Spam.append(f.read())

randomState = np.random.RandomState(seed)
randomState.shuffle(All_Ham)
randomState.shuffle(All_Spam)

training_data = All_Ham[:number_of_training_emails] + All_Spam[:number_of_training_emails]
testing_data = All_Ham[number_of_training_emails:] + All_Spam[number_of_training_emails:]


# important: use .split() to separate the words
# This function builds the vocabulary and calculates the word frequencies
def mvb(data):
    vocabulary = set()

    for email in data:
        words = set(email.split())
        vocabulary.update(words)

    word_counter = {word: {'spam': 0, 'non_spam': 0} for word in vocabulary}

    # Counts word occurrences in spam and non-spam emails
    for email in training_data:
        words = set(email.split())
        is_spam = email in All_Spam[:number_of_training_emails]
        for word in words:
            if is_spam:
                word_counter[word]['spam'] += 1
            else:
                word_counter[word]['non_spam'] += 1

    # Calculates probabilities with Laplace smoothing
    total_spam = len(All_Spam[:number_of_training_emails])
    total_non_spam = len(All_Ham[:number_of_training_emails])

    # Creates a dictionary that houses the probability of a word
    result = {
        word: {
            'spam': (word_counter[word]['spam'] + 1) / (total_spam + 2),
            'non_spam': (word_counter[word]['non_spam'] + 1) / (total_non_spam + 2)
        }
        for word in vocabulary
    }

    return result


# This function classifies an email as spam or non-spam

def classify(email, probabilities):
    spam_value = 0
    non_spam_value = 0
    spam_likely = False
    for word in email.split():
        if word in probabilities:
            spam_value += np.log(probabilities[word]['spam'])
            non_spam_value += np.log(probabilities[word]['non_spam'])
    # If the calculated spam probability is greater, classify as spam
    if spam_value > non_spam_value:
        spam_likely = True

    return spam_likely


# This function evaluates model performance
def evaluations(result):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Compares model predictions with the actual labels
    for email in testing_data:

        is_spam_prediction = classify(email, result)

        is_spam_actual = email in All_Spam[number_of_training_emails:]

        if is_spam_actual and is_spam_prediction:
            true_positive += 1
        elif not is_spam_actual and not is_spam_prediction:
            true_negative += 1
        elif not is_spam_actual and is_spam_prediction:
            false_positive += 1
        elif is_spam_actual and not is_spam_prediction:
            false_negative += 1

    # returns the different values from the diagram
    return true_positive, true_negative, false_positive, false_negative


# Calculates the precision

def precision(true_positive, false_positive):
    return (true_positive) / (true_positive + false_positive)


# Calculates the recall metric
def recall(true_positive, false_negative):
    return (true_positive) / (true_positive + false_negative)


# Calculates the true negative rate
def true_negative_rate(true_negative, false_positive):
    return (true_negative) / (true_negative + false_positive)


# Calculates the f1 score
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# Calculates the accuracy
def accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


# main function
if __name__ == '__main__':
    result = mvb(training_data)
    # print(testing_data)
    true_positive, true_negative, false_positive, false_negative = evaluations(result)
    print(true_positive, true_negative, false_positive, false_negative)

    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    trn = true_negative_rate(true_negative, false_positive)
    f1 = f1_score(prec, rec)
    acc = accuracy(true_positive, true_negative, false_positive, false_negative)

    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'True negative rate: {trn}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {acc}')

# The results when I ran my program and my analysis
"""
true positive: 100 
true negative: 33
false positive: 67
false negative: 0
Precision: 0.5988023952095808
Recall: 1.0
True negative rate: 0.33
F1 Score: 0.7490636704119851
Accuracy: 0.665

Based on the results of the model and how it is being evaluated, we can see that it is decent at identifying spam emails.
Though when it comes to correctly highlighting non_spam emails, it is not as successful as seen in the true negative rate of .33.
This claim is again backed up by the precision of about .599. This tells us that when a model predicts ane mail is spam, it is only correct about
60% of the time. 

The model has a high true positive rate, making it very effective at identifying spam emails. But, if we look at the false negatives and its absence
of false negatives, we can see that that model didn't label any spam emails as non-spam. This is highlighted in the recall with it being 1.
The true negative rate is low with only 33 non-spam emails correctly labeled as non-spam. A high number of false positives shows that 67 non-spam emails were also incorrectly
labeled as spam. 
The model is clearly over-predicting spam emails, which identifies bias towards classifying emails as spam. The model could also be seeing common words in both
the spam and non-spam emails which can create some confusion. This can be overcome by increasing the quantity of the training set as a high number of spam and ham emails can 
yield more variety in the type of words used. I would also advise to even consider a 'common word' filter that as well. 
"""
