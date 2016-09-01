# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:17:35 2016

POS tag list:
CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

In order to chunk, we combine the part of speech tags with 
regular expressions. Mainly from regular expressions, 
we are going to utilize the following:
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line

NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian



@author: ceciliaLee
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


## using the stop_words set to remove the stop words from your text:
def stopWords_remove(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    filtered_sent=[w for w in word_tokens if not w in stop_words]
    return filtered_sent
    


## Using Porter stemmer to stemming
def portweStemm(text):
    
    result=[]
    ps = PorterStemmer()
    for w in text:
        result.append(ps.stem(w))
    return result
        

## Part-of-Speech Tagging with TNLK
## the sentence tokenizer - PunktSentenceTokenizer is capable of unsupervised machine learning, 
# so you can actually train it on any body of text that you use
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk
## The output should be a list of tuples, where the first element in the tuple is the word, 
## and the second is the part of speech tag
def pos(train_text, sample_text):
    ## Train the Punkt tokenizer
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text) ## tokenize
   ## run through and tag all of the parts of speech per sentence
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            return tagged
    except Exception as e:
        print str(e)
    
##Chunking -  group into what are known as "noun phrases."
## that is: grouping nouns with the words that are in relation to them
# the part of speech tags are denoted with the "<" and ">" and we 
# can also place regular expressions within the tags themselves, 
# so account for things like "all nouns" (<N.*>)
    
# Chunking is based on part of POS
def chunking(train_text, sample_text):
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged =nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            ## <RB.?>* = "0 or more of any tense of adverb,"
            ## <VB.?>* = "0 or more of any tense of verb,"
            ## <NNP>+ = "One or more proper nouns,"
            ## <NN>? = "zero or one singular noun."
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged) ## "chunked" variable is an NLTK tree
            
            print chunked
            ## if we want to get just the chunks, ignoring the rest
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print subtree
                
            chunked.draw()     
    except Exception as e:
        print str(e)
    

## Named Entity Recognition with NLTK
def namedEntity(train_text, sample_text):
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            namedEnt=nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()
        return namedEnt
    except Exception as e:
        print str(e)
        
## Lemmatizing with NLTK
## lemmatize takes a part of speech parameter, "pos." If not supplied, the default is "noun." 
from nltk.stem import WordNetLemmatizer
def lemmatizing(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)
    
## Using WordNet in NLTK 
from nltk.corpus import wordnet
def synsets(word):
    syns=wordnet.synsets(word)  # search for synstes
    print syns[0].name()   # An example of a synset
    print syns[0].lemmas()[0].name()  ## Just the word
    print syns[0].definition()  ## definition of the 1st synset
    print syns[0].examples()   #examples of the word in use
    
    ## find the synonyms and antonymsnof a given lemmas
    synonums=[]
    antonyms=[]
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonums.append(i.name())
            if i.antonyms():
                antonyms.append(i.antonyms()[0].name())
    return synonums, antonyms
    
## use WordNet to compare the similarity of two words and their tenses, 
#  by incorporating the Wu and Palmer method for semantic related-ness.    
def compSim(word1, word2):
    w1 = wordnet.synset(word1+'.n.01')
    w2 = wordnet.synset(word2+'.n.01')
    sim=w1.wup_similarity(w2)
    return sim
    



text = 'This is a sample sentence, showing off the stop words filtration.'
filSen=stopWords_remove(text)
print '1. stop words removal e.g.:',filSen
print 
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
print '2. Stemming e.g.:', portweStemm(example_words)
print 
## POS
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
tagged=pos(train_text, sample_text)
print '3. POS e.g.: ', tagged
print 
## Chunking
print '4. Chunking e.g.: '
chunking(train_text, sample_text)

## Named Entity
namedEnt=namedEntity(train_text, sample_text)
print '5. Named Entity e.g.: ', namedEnt 

## WordNet in NLTK 
print '6. WordNet, synomus and antonyms, make comparison: '
word='good'
syn, anto=synsets(word)
print 'synomus and antonyms: ', syn, anto
word1, word2='ship', 'boat'
sim=compSim(word1, word2)
print 'Similarity: ', sim
