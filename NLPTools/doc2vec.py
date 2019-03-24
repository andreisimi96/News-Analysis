#Andrei Simion 2018

from os import listdir
from os.path import isfile, join
import gensim
import gensimAux
import re
from preprocessing import *
import numpy as np
from scipy.spatial.distance import cosine
from gensimAux import lemmaDict, stopwords
from articleWrapper import loadArticles

#doc2vec params
windowSize = 6
vectorSize = 200


class doc2VecWrapper():
    def __init__(self):
        self.docList = []
        self.docLabels = []
        self.d2vModel = None
        #check C compiler for speedy results
        #consider using anaconda environment if this returns != 1
        print(gensim.models.doc2vec.FAST_VERSION)

    def initDoc2VecInput(self, trainFolders, articleTrainFolder):
        #trainFolders = [trainFolder for trainFolder in trainFolders]
    
        self.docLabels = []
        self.docList = []
        splitString = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)]

        #load the corpora from the trainFolders, not including the news articles
        #this part is done as we most likely don't have enough news articles to
        #train the doc2vec well
        for trainFolder in trainFolders:

            #gensim doesn't support documents with over 10.000 tokens, so we work around that limitation
            preChopFolderLabels = [f for f in listdir(trainFolder) if isfile(join(trainFolder, f))]
            postChopFolderLabels = []

            for index in range(len(preChopFolderLabels)):
                doc = open(trainFolder + preChopFolderLabels[index], encoding = "utf8").read()
                docChops = splitString(doc, 40000)

                for i in range(len(docChops)):
                    postChopFolderLabels.append(preChopFolderLabels[index] + str(i))
                self.docList += docChops

            self.docLabels += postChopFolderLabels

        #load the articles from folders
        articleList, articleLabels = loadArticles(articleTrainFolder)

        #add them to self.docLabels and self.docList to increase train corpus size
        self.docLabels += articleLabels
        self.docList += articleList

        print("finished taking docs, starting training")

    #mode can be either dm(1) or dbow(2), 
    def trainDoc2Vec(self, trainFolders, articleTrainFolder, savePath, mode):
        self.initDoc2VecInput(trainFolders, articleTrainFolder)

        iterator = gensimAux.TaggedLineDocument(self.docList, self.docLabels)
        
        self.d2vModel = gensim.models.Doc2Vec(vector_size=vectorSize, dm=mode, window=windowSize, min_count=1, workers=6, dbow_words=1)
        self.d2vModel.build_vocab(iterator)
        print("finished building vocab")

        self.d2vModel.train(iterator, total_examples = len(self.docList), epochs = 24)
        print("finished training model")

        self.d2vModel.save(savePath + "doc2vec_model")

    def loadDoc2VecModel(self, modelPath):
        self.d2vModel = gensim.models.doc2vec.Doc2Vec.load(modelPath)

    #input as 2 strings
    #for testing only, you can ignore this
    def compareDocs(self, doc1, doc2):
        doc1 = preprocessText(doc1, gensimAux.lemmaDict, gensimAux.stopwords)
        doc2 = preprocessText(doc2, gensimAux.lemmaDict, gensimAux.stopwords)
        
        #infer vectors
        vectDoc1 = self.d2vModel.infer_vector(doc1, steps = 30)
        vectDoc2 = self.d2vModel.infer_vector(doc2, steps = 30)

        print(self.d2vModel['www.realitatea.net1369'])
        print(1 - cosine(vectDoc1, self.d2vModel['www.realitatea.net1369']))

        vectDoc1 = list(vectDoc1)
        vectDoc2 = list(vectDoc2)
        vectDoc1 += [0.1 for i in range(150)]
        print(1 - len(vectDoc1))
        vectDoc2 += [0.9 for i in range(150)]

        vectDoc1 = np.asarray(vectDoc1)
        vectDoc2 = np.asarray(vectDoc2)
        #print(len(vectDoc1))

        #normalize cosine similarity to 0, 1
        normalize = 0
        dist = 1 - cosine(vectDoc1, vectDoc2)

        return dist


    #use the trained word vectors in doc2vec models
    #to create auxiliary document embeddings 
    #this outputs embeddings in a list format
    #WARNING: use only models trained with dm = 1
    #WARNING: dm = 0, using DBOW model, leaves the word embeddings random
    #WARNING: training only the document embeddings
    def createDocumentEmbeddingW2V(self, text, lemmaDict, stopwords):
        tokens = preprocessText(text, lemmaDict, stopwords)

        #convert each token to vector representation
        #if the model doesn't contain the word, won't work AT ALL 
        tokenVectors = [self.d2vModel[token] for token in tokens]
        return list(np.mean(tokenVectors, axis=0).round(decimals=6))

if __name__ == "__main__":

    trainFolders = ["doc2vec_train_data/eseuri_gimnaziu/",
                    "doc2vec_train_data/txt/",
                    "doc2vec_train_data/txt-par/"]
    trainFolders = ["doc2vec_train_data/txt-par/"]
    articleTrainFolder = "C:/textePublicatii/"
    testFolder = "testbench/"

    #start lemmas & stopwords
    lemmaDict = initLematizer()
    stopwords = createStopwordsSet()

    #doc2vec path for dm = 1, distributed memory model used
    dmD2VModelPath = 'doc2vec_models/classic, dm + dm_concat = 0/'
    
    #doc2vec path for dm = 0, distributed bag of words model used
    dbowD2VModelPath = 'doc2vec_models/dbow + dm_concat = 0/'

    wrapper = doc2VecWrapper()
    wrapper.loadDoc2VecModel(dbowD2VModelPath + "doc2vec_model")
    
    #wrapper.trainDoc2Vec(trainFolders, articleTrainFolder, savePath="TODO")