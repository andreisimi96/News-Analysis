#Andrei Simion 2018

#add paths
import sys
sys.path.append("./NLPTools")

from articleWrapper import *
from doc2vec import doc2VecWrapper
from LDA import LDAWrapper


#ways to train 
dbow = 0
dm = 1


#pipeline of the application, comment/uncomment what you need/see fit

def doc2Vec(articleTrainFolder):
    auxTrainFolders = ["doc2vec_train_data/txt-par/",
                    "doc2vec_train_data/TextePublicatiiVara/agerpres.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/hotnews.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/adevarul.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/ziare.com/",
                    "doc2vec_train_data/TextePublicatiiVara/stirileprotv.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/digi24.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/evz.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/b1.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/romaniatv.ro/",
                    "doc2vec_train_data/TextePublicatiiVara/realitatea.net/",
                    "doc2vec_train_data/TextePublicatiiVara/mediafax.ro/"]
    auxTrainFolders = []
    """doc2vec: either train or load"""
    d2vWrapper = doc2VecWrapper()

    #doc2vec path for dm = 0, distributed bag of words model used
    dmD2VModelPath = 'doc2vec_models/classic, dm + dm_concat = 0/'
    dbowD2VModelPath = 'doc2vec_models/dbow + dm_concat = 0/'

    d2vWrapper.trainDoc2Vec(auxTrainFolders, articleTrainFolder, dmD2VModelPath, dm)
    #d2vWrapper.loadDoc2VecModel()

def lda(articleTrainFolder):
    """lda: do not use train & load at the same time"""
    lda = LDAWrapper(articleTrainFolder)
    lda.trainLDA()
    lda.visualize()
    #lda.loadLDAModel()
    topics = lda.getTopicsForModel(topN=10)

    for topic in topics:
        print(topic)
        print("")

    #lda.visualize()
    #lda.generateTopicFiles(articleTrainFolder)

if __name__ == "__main__":
    import os
    print(os.path.dirname(os.path.realpath(__file__)))
    
    articleTrainFolder = "C:/textePublicatiiRelevante/"
    
    """article filtering, crawler might have missed some cases"""
    #filterArticles(articleTrainFolder)
    #cleanArticleFolder(articleTrainFolder)

    """processing steps"""
    #lda(articleTrainFolder)
    #doc2Vec(articleTrainFolder)

