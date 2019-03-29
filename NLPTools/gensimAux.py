#Andrei Simion 2018

import gensim
from preprocessing import preprocessText, initLematizer, createStopwordsSet
import os
from articleWrapper import *

#provides an easy way of iterating through the documents
#and returning gensim objects
#WARNING: the articles MUST be pairs of files with .txt and .info extensions

"""Doc2Vec part"""
TaggedDocument = gensim.models.doc2vec.TaggedDocument

#cache lemmaDict and stopwordsSet
lemmaDict = initLematizer()
stopwords = createStopwordsSet()

class TaggedLineDocument(object):
    def __init__(self, docList, docLabels):
       self.labels_list = docLabels
       self.doc_list = docList

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):

            tokens = preprocessText(doc, lemmaDict, stopwords)
            
            if len(tokens) > 40:
                yield TaggedDocument(preprocessText(doc, lemmaDict, stopwords), [self.labels_list[idx]])


"""LDA/tfidf part"""
#iterator for documents
def iterDocs(path):
    #get directories that contain the articles
    dirs = getArticleDirs(path)

    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                article = dir + "/" + os.path.splitext(file)[0]
                
                title, _, _, publication = parseArticleInfoFile(article)

                article = open(article + ".txt", encoding = "utf8").read()
                tokens = preprocessText(article, lemmaDict, stopwords)

                #skip very short articles
                if len(tokens) > 40:
                    yield title, tokens


def createDictLDA(path):
    #create dictionary for words and no of appearances
    docStream = (tokens for _, tokens in iterDocs(path))
    id2wordArticles = gensim.corpora.Dictionary(docStream)
    id2wordArticles.filter_extremes(no_below = 20, no_above = 0.1)
    print("finished filtering")

    return id2wordArticles




