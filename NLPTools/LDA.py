#Andrei Simion 2018

import gensim
from gensimAux import lemmaDict, stopwords, iterDocs, createDictLDA
from preprocessing import *
import pyLDAvis
import pyLDAvis.gensim
import os
from articleWrapper import getArticleDirs


class LDAWrapper():
    def __init__(self, path):
        self.ldaModel = None

        self.dictionary = createDictLDA(path)
        print('finished dict creation')
        
        self.path = path

        self.docStream = (tokens for _, tokens in iterDocs(path))

        self.corpus = [ self.dictionary.doc2bow(doc) for doc in self.docStream ]
        print('finished corpus creation')
        

    #trains LDA model and saves it to disk
    def trainLDA(self):
        print(os.getcwd())
        self.ldaModel = gensim.models.LdaModel(self.corpus, num_topics = 100, alpha='asymmetric', id2word = self.dictionary, passes = 50)
        self.ldaModel.save('lda_model')
        
    #visualization with pyLDAvis map
    def visualize(self):
        print(os.getcwd())
        p = pyLDAvis.gensim.prepare(self.ldaModel, self.corpus, self.dictionary)
        pyLDAvis.save_html(p, 'lda.html')

    #loads pre-trained LDA model from disk
    def loadLDAModel(self):
        print(os.getcwd())
        self.ldaModel = gensim.models.LdaModel.load('lda_model')


    #prints the topics for the entire model
    def getTopicsForModel(self, topN):
        modelTopics = self.ldaModel.top_topics(corpus=self.corpus, dictionary=self.dictionary, topn=topN)
        return modelTopics


    #gets topics from a documents in a file and prints them in a .tpcs extension
    #skips topics with probability lower than minProbabil
    def getTopicsForArticle(self, article, minProbabil):

        bow = open(article, encoding = "utf8").read()
        bow = preprocessText(bow, lemmaDict, stopwords)
        bow = self.dictionary.doc2bow(bow)

        docTopics = self.ldaModel.get_document_topics(bow, minimum_probability = minProbabil)
        #sort by distribution
        docTopics.sort(key=lambda tup: tup[1], reverse = True)


        output = os.path.splitext(article)[0] + ".tpcs"
        outfp = open(output, mode = 'w', encoding="utf8")

        for (topicId, distrib) in docTopics:

            outfp.write(str(distrib) + "\n")
            for (word, distrWord) in self.ldaModel.show_topic(topicId):
                outfp.write(word + " " + str(distrWord) + " ")
            outfp.write("\n")
        
        outfp.close()

    def generateTopicFiles(self, articlePath):
        dirs = getArticleDirs(articlePath)

        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith(".txt"):
                    fullPath = dir + "/" + file
                    self.getTopicsForArticle(fullPath, 0.05)

if __name__ == "__main__":
    path = "C:/textePublicatii/"

    lda = LDAWrapper(path)
    #lda.loadLDAModel()
    #lda.trainLDA()
    
    dirs = getArticleDirs(path)

    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                fullPath = dir + "/" + file
                lda.getTopicsForArticle(fullPath, 0.05)



    #lda.trainLDA()
    #lda.visualize()

    #trainLDA(path)
    #lemmaDict = initLematizer()
    #stopwords = createStopwordsSet()
    #visualize()
    #print(removeDiacritics("sâmbătă"))