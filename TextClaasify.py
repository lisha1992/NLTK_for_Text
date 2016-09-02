# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:08:36 2016

@author: ceciliaLee
"""

import nltk


## Text Classification with TNLK
import random
from nltk.corpus import movie_reviews ## use the movie reviews database
def textClassifier(documents):
    random.shuffle(documents) ## training or testing
   # print documents[1]
    
    all_words=[]
    for w in movie_reviews.words():
        all_words.append(w.lower())
        
    ## Frequency distribution, to then find out the most common words
    all_words_fre=nltk.FreqDist(all_words) ## return a list of tuple with the word and its frequency
    
    print 'Top 15 most common words: ', all_words_fre.most_common(15)
    print all_words_fre['stupid']
    return all_words_fre
    
    
## Converting words to Features with NLTK
## based on the frequency distribtion of words
def word2Feac(documents, doc):## documents, doc are 2 different documents
    all_words_fre = textClassifier(documents)
    word_features = list(all_words_fre.keys())[:3000] ## The top 3000 most common words
    words=set(doc)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features  ## a parse vector
 

def bayesClassifier(features):
    ## split the data into training set and testing set
    training_set=features[:1900]
    testing_set=features[1900:]
    ## Train the classifier
    classifier=nltk.NaiveBayesClassifier.train(training_set)
    ## using testing set to test the accuracy of the classifier
    accurate=nltk.classify.accuracy(classifier, testing_set)*100
    
    ##see what the most valuable words are when it comes to positive or negative reviews
    classifier.show_most_informative_features(15)
    return accurate
    
    

   
     

documents=[(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
print '1. Text Classification Ex: '                 
all_words_fre = textClassifier(documents)    
print 
print '2. Convert words to Features Ex: ' 
doc='neg/cv000_29416.txt'
featureset = word2Feac(documents, movie_reviews.words(doc))
print 'feature set: ', featureset
print type(featureset)
print
print '3. Naive Bayes Classifier Ex: ' 
## The testing is UNFINISHED
temp=[]
featurelist=[]
for key, value in featureset.iteritems():
    temp=(key, value)
    featurelist.append(temp)

# accurate=bayesClassifier(featurelist)

             
                