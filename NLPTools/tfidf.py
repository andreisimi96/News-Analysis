#Andrei Simion 2018

import gensim
from gensimAux import lemmaDict, stopwords, iterDocs, createDictLDA
from preprocessing import *
import os
from articleWrapper import getArticleDirs


class tfidfWrapper():
    def __init__(self, path):
        self.tfidfModel = None

        self.dictionary = createDictLDA(path)
        print('finished dict creation')

        self.path = path
        

    #trains tfidf and saves it to disk
    def trainTfidf(self):
        self.docStream = (tokens for _, tokens in iterDocs(self.path))
        self.corpus = [ self.dictionary.doc2bow(doc) for doc in self.docStream ]
        print('finished corpus creation')

        print(os.getcwd())
        self.tfidfModel = gensim.models.TfidfModel(self.corpus, id2word = self.dictionary)
        self.tfidfModel.save('tfidf_model')

    #loads pre-trained tfidf model from disk
    def loadTfidfModel(self):
        print(os.getcwd())
        self.tfidfModel = gensim.models.TfidfModel.load('tfidf_model')

    def getKeywordsForArticle(self, article):

        bow = open(article, encoding = "utf8").read()
        bow = preprocessText(bow, lemmaDict, stopwords)
        bow = self.dictionary.doc2bow(bow)

        vector = self.tfidfModel[bow]
        vector.sort(key=lambda tup: tup[1])

        keywords = []
        for i in range(len(vector) - 1, 0, -1):
            if (i < (len(vector) - 11)):
                continue
            else:
                keywords.append(self.dictionary[vector[i][0]])

        #print(keywords)

        return keywords
       

if __name__ == "__main__":
    path = "C:/textePublicatii/"

    tfidf = tfidfWrapper(path)
    #tfidf.trainTfidf()
    tfidf.loadTfidfModel()
    #tfidf.getKeywordsForArticle("C:/textePublicatii/antena3.ro/0.txt")