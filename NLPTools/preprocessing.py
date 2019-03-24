#Andrei Simion 2018

from nltk.tokenize import RegexpTokenizer
import time 
from contextlib import contextmanager
from timeit import default_timer
import unicodedata

#mainly used as a normalization method between different publications
#as some do not use diacritics, applied after text.lower()
def removeDiacritics(text):
    diacritics = {'ă' : 'a',
                 'â' : 'a',
                 'î' : 'i',
                 'ș' : 's',
                 'ț' : 't',
                 'ţ' : 't',
                 'ş' : 's'}
    return ''.join(diacritics.get(ch, ch) for ch in text)

def removePunctuationAndLowercase(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(removeDiacritics(text.lower()))
    return tokens

def removeNumbers(tokens):
    return [token for token in tokens if not token.isdigit()]

def initLematizer():
    with open("lemmatization-ro.txt", encoding = "utf8") as f:
        content = f.readlines()
        lemmaDict = {}
        for line in content:
            line = removeDiacritics(line)
            splitLine = line.split()
            lemmaDict[splitLine[1]] = splitLine[0]
    return lemmaDict

def createStopwordsSet():
    with open("stopwords-ro.txt", encoding = "utf8") as f, \
         open("stopwords-en.txt", encoding = "utf8") as g:
        lines = f.readlines()
        stopwords = [removeDiacritics(sw.rstrip('\n')) for sw in lines]
        lines = g.readlines()
        stopwords += [removeDiacritics(sw.rstrip('\n')) for sw in lines]

    #convert to set for better performance
    stopwords = set(stopwords)
    return stopwords

def removeStopwords(tokens, stopwords):
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

#using the lemma map(doesn't use POS tagging), replace words
def lemmatize(tokens, lemmaDict):
    for index, token in enumerate(tokens):
        if token in lemmaDict:
            tokens[index] = lemmaDict[token]
    return tokens

def preprocessText(text, lemmaDict, stopwords):
     tokens = removePunctuationAndLowercase(text)
     tokens = removeNumbers(tokens)
     tokens = lemmatize(tokens, lemmaDict)
     tokens = removeStopwords(tokens, stopwords)
     return tokens