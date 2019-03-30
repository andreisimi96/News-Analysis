#Andrei Simion 2018
import gensim
from scipy.spatial.distance import cosine
from elasticSearchWrapper import createESDict, extractData
import numpy as np
from itertools import combinations
from doc2vec import doc2VecWrapper
from preprocessing import createStopwordsSet, initLematizer


def compareVectors(vector1, vector2):
    return (1 - cosine(np.asarray(vector1), np.asarray(vector2)))


#for Bradley Terry input
publicationsDict = {
                        "agerpres.ro" : '1',
                        "hotnews.ro" :  '2',
                        "adevarul.ro" : '3',
                        "ziare.com" : '4',
                        "stirileprotv.ro" : '5',
                        "digi24.ro" : '6',
                        "b1.ro" : '7',
                        "romaniatv.net" : '8', 
                        "realitatea.net" : '9',
                        "mediafax.ro" : '10',
                        "libertatea.ro" : '11',
                        "antena3.ro" : '12'
                    }


#outputs in file the pairwise similarity between the documents
def compareDocuments(modelName, similarityOutput):
    d2vModel = gensim.models.doc2vec.Doc2Vec.load(modelName)
    esDict = createESDict()

    txtSimilarity = open(similarityOutput,'w', encoding='utf8')
    
    i = 0

    #a day and a half
    period = 60 * 60 * 24 * 1.5

    for label1, label2 in combinations(d2vModel.docvecs.doctags, r=2):
        #only care if the labels are in the esDict 
        #and if the publications are different
        if (label1 in esDict) and (label2 in esDict) and \
            (esDict[label1][1] != esDict[label2][1]):
            i += 1
            if i % 500000 == 0:
                print(i)

            pubDate1 = esDict[label1][2]
            pubDate2 = esDict[label2][2]

            if abs(pubDate1 - pubDate2).total_seconds() < period:
                similarity = compareVectors(d2vModel[label1], d2vModel[label2])
                txtSimilarity.write(label1 + " " + label2 + " " + str(similarity) + "\n")

    txtSimilarity.close()

def compareDocumentsW2V(dmD2VModelPath, similarityOutput):
    wrapper = doc2VecWrapper()
    wrapper.loadDoc2VecModel(dmD2VModelPath)

    #init lemmas & stopwords
    lemmaDict = initLematizer()
    stopwords = createStopwordsSet()
    esDict = createESDict()

    txtSimilarity = open(similarityOutput,'w', encoding='utf8')
    
    #a day and a half
    period = 60 * 60 * 24 * 1.5

    #memorize word embeddings per label
    wordEmbedding = {}
    for label in wrapper.d2vModel.docvecs.doctags:
        if (label in esDict) :
            content = esDict[label][3]
            wordEmbedding[label] = wrapper.createDocumentEmbeddingW2V(content, lemmaDict, stopwords)
    print("finished generating word embeddings for documents")
    
    print("starting pairwise comparison")
    i = 0
    for label1, label2 in combinations(wrapper.d2vModel.docvecs.doctags, r=2):

        #only care if the labels are in the esDict 
        #and if the publications are different
        if (label1 in esDict) and (label2 in esDict) and \
            (esDict[label1][1] != esDict[label2][1]):
            i += 1
            if i % 500000 == 0:
                print(i)

            pubDate1 = esDict[label1][2]
            pubDate2 = esDict[label2][2]

            if abs(pubDate1 - pubDate2).total_seconds() < period:
                similarity = compareVectors(wordEmbedding[label1], wordEmbedding[label2])
                txtSimilarity.write(label1 + " " + label2 + " " + str(similarity) + "\n")

    txtSimilarity.close()

#returns no of articles which are unique per publication
def getMostUniquePublications(modelName, similarityThreshold):

    d2vModel = gensim.models.doc2vec.Doc2Vec.load(modelName)
    esDict = createESDict()

    #retain unique articles and total no of articles
    uniquePublicationArticles = {}
    publicationArticles = {}

    for label in d2vModel.docvecs.doctags:
        if label in esDict:

            #add to total articles
            if esDict[label][1] in publicationArticles:
                publicationArticles[esDict[label][1]] += 1
            else:
                publicationArticles[esDict[label][1]] = 1

            mostSimilarArticles = d2vModel.docvecs.most_similar(label)
            #filter mostSimilar (i.e. ignore most similar from same publication
            #or those that aren't even indexed in elastic)
            mostSimilarArticles = [x for x in mostSimilarArticles if x[0] in esDict and esDict[x[0]][1] != esDict[label][1]]

            #check first most similar and it's cosine similarity
            if len(mostSimilarArticles) == 0 or mostSimilarArticles[0][1] < similarityThreshold:
                if esDict[label][1] in uniquePublicationArticles:
                    uniquePublicationArticles[esDict[label][1]] += 1
                else:
                    uniquePublicationArticles[esDict[label][1]] = 1

    return uniquePublicationArticles, publicationArticles

def getRelevantArticles(modelName, similarityThreshold):
    d2vModel = gensim.models.doc2vec.Doc2Vec.load(modelName)
    esDict = createESDict()

    relevantArticles = []
    for label in d2vModel.docvecs.doctags:
        if label in esDict:

            mostSimilarArticles = d2vModel.docvecs.most_similar(label)
            #filter mostSimilar (i.e. ignore most similar from same publication
            #or those that aren't even indexed in elastic)
            mostSimilarArticles = [x for x in mostSimilarArticles if x[0] in esDict and esDict[x[0]][1] != esDict[label][1]]

            #check first most similar and it's cosine similarity
            if len(mostSimilarArticles) == 0 or mostSimilarArticles[0][1] < similarityThreshold:
                continue
            else:
                publication = esDict[label][1]
                path = esDict[label][1] + "/" + label[len(publication):]
                relevantArticles.append(path)

    return relevantArticles

def copyRelevantArticles(relevantArticles):
    from shutil import copy2
    import errno
    import os

    print("Starting")


    for relevantArticle in relevantArticles:

        src = "C:/textePublicatii/" + relevantArticle
        dest = "C:/textePublicatiiRelevante/" + relevantArticle

        try:
            copy2(src + ".txt", dest + ".txt")
            copy2(src + ".info", dest + ".info")
            copy2(src + ".ent", dest + ".ent")
            copy2(src + ".tpcs", dest + ".tpcs")

        except IOError as e:
            # ENOENT(2): file does not exist, raised also on missing dest parent dir
            if e.errno != errno.ENOENT:
                raise
            # try creating parent directories
            os.makedirs(os.path.dirname(dest))
            copy2(src + ".txt", dest + ".txt")
            copy2(src + ".info", dest + ".info")
            copy2(src + ".ent", dest + ".ent")
            copy2(src + ".tpcs", dest + ".tpcs")

def createBTInput(timeThreshold, similarityThreshold, similarityFile = 'similarity.txt', rankingOutput = 'ranking.csv'):
    csvOutput = open(rankingOutput,'w', encoding='utf8')
    esDict = createESDict()

    i = 0
    for line in open(similarityFile, encoding='utf8'):
        if i % 500000 == 0:
            print(i)
        i += 1
        label1, label2, similarity = line.split()
        similarity = float(similarity)

        publication1 = esDict[label1][1]
        publication2 = esDict[label2][1]


        #ignore if the similar articles come from the same publication
        #shouldn't fall through this too many times :)
        if publication1 == publication2:
            continue

        pubDate1 = esDict[label1][2]
        pubDate2 = esDict[label2][2]

        #filter using thresholds
        if abs(pubDate1 - pubDate2).total_seconds() < timeThreshold \
            and similarity > similarityThreshold:

            winner = 1 if pubDate1 > pubDate2 else 2
            csvOutput.write(publicationsDict[publication1] + "," + publicationsDict[publication2] + "," + str(winner) + "\n")

    csvOutput.close()

if __name__ == "__main__":
    day = 60 * 60 * 24
    hrs6 = day / 4
    hrs8 = day / 3
    hrs12 = day / 2


    timeThreshold = hrs8
    similarityThreshold = 0.8

    dbowD2VModelPath = 'doc2vec_models/dbow + dm_concat = 0/'
    dmD2VModelPath = 'doc2vec_models/classic, dm + dm_concat = 0/'
    similarityOutputW2V = 'BT_Similarity/similarity_w2v.txt'
    similarityOutputD2V1 = 'BT_Similarity/similarity_d2v_dbow.txt'
    similarityOutputD2V2 = 'BT_Similarity/similarity_d2v_dm.txt'

    import time
    start = time.time()

    #decomment only if you want similarity files to be updated
    #WARNING: they are very big
    """compareDocuments(dbowD2VModelPath + 'doc2vec_model', similarityOutputD2V1)
    compareDocuments(dmD2VModelPath + 'doc2vec_model', similarityOutputD2V2)
    compareDocumentsW2V(dmD2VModelPath + 'doc2vec_model', similarityOutputW2V)"""

    #createBTInput(timeThreshold, similarityThreshold, similarityOutputD2V1, 'ranking_doc2vec_dbow.csv')
    #createBTInput(timeThreshold, similarityThreshold, similarityOutputD2V2, 'ranking_doc2vec_dm.csv')
    #createBTInput(timeThreshold, similarityThreshold, similarityOutputW2V, 'ranking_word2vec_cbow.csv')
    """uniquePublicationArticles, publicationArticles = \
                    getMostUniquePublications(dbowD2VModelPath + 'doc2vec_model', 0.7)

    for k in publicationArticles:
        print(k + " " + str(1 - float(uniquePublicationArticles[k] / publicationArticles[k])))"""

    copyRelevantArticles(getRelevantArticles(dbowD2VModelPath + 'doc2vec_model', 0.8))

    end = time.time()
    print(end - start)

    """uniquePublicationArticles, publicationArticles = getMostUniquePublications(dmD2VModelPath + "doc2vec_model", similarityThreshold)
    for x in uniquePublicationArticles:
        print(x + " " + str(uniquePublicationArticles[x]) + "/" + publicationArticles[x])"""