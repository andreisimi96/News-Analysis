#Andrei Simion 2018

import os, errno
import operator
import hashlib
from preprocessing import *
from difflib import SequenceMatcher


#get directories that contain articles
def getArticleDirs(path):
    return [path + dir for dir in os.listdir(path)]

#takes article path without extension and
#returns (article title, publishing date, link, publication"
def parseArticleInfoFile(article):
    info = open(article + ".info", encoding = "utf8")
    infoLines = info.readlines()

    linesNo = len(infoLines)
    publication = infoLines[linesNo - 1].replace('\n', '')
    link = infoLines[linesNo - 2].replace('\n', '')
    publishingDate = infoLines[linesNo - 3].replace('\n', '')

    #hotfix for bug that prints title with "\n" instead of " "
    title = "".join([titlePiece for titlePiece in infoLines[:linesNo - 3]])
    title = title.replace('\n', ' ')

    return title, publishingDate, link, publication

#takes article path without extension and
#returns (persons, locations, organizations)
def parseArticleEntityFile(article):
    ent = open(article + ".ent", encoding = "utf8")
    entLines = ent.readlines()

    #readlines also keeps the newline character, replace it
    #also, convert to set, we don't care about duplicates
    entLines = set([line.replace('\n', '') for line in entLines])

    persons = []
    locations = []
    organizations = []

    for line in entLines:
        line = line.split(maxsplit=2)
        if line[0] == 'I-LOC':
            locations.append(line[1])
        elif line[0] == 'I-PER':
            persons.append(line[1])
        elif line[0] == 'I-ORG':
            organizations.append(line[1])
    
    return persons, locations, organizations

#takes article path without extension and integer n
#and returns the n keywords
def parseArticleTopicFile(article, topKeywordsN):
    tpc = open(article + ".tpcs", encoding = "utf8")
    tpcLines = tpc.readlines()

    wordDistributions = {}


    for index in range(1, len(tpcLines), 2):
        topicWords = tpcLines[index].split()

        for i in range(0, len(topicWords)-1, 2):
            if topicWords[i] not in wordDistributions:
                wordDistributions[topicWords[i]] = float(topicWords[i+1]) * float(tpcLines[index-1])
            else:
                wordDistributions[topicWords[i]] += float(topicWords[i+1]) * float(tpcLines[index-1])


    sortedDistributions = sorted(wordDistributions.items(), key=operator.itemgetter(1), reverse=True)
    topWords = [x[0] for x in sortedDistributions]
    topWords = topWords[:topKeywordsN]

    return topWords


#returns docLabels and docs for articles
#mainly for the purpose of doc2vec
def loadArticles(articleTrainFolder):
    articleLabels = []
    articleDocs = []

    #get directories that contain the articles
    dirs = getArticleDirs(articleTrainFolder)
    print("There's a total of: " + str(len(dirs)) + " news publications")

    for dir in dirs:
        print(len(os.listdir(dir)))
        for file in os.listdir(dir):
   
            if file.endswith(".txt"):
                article = dir + "/" + os.path.splitext(file)[0]
                articleDoc = open(article + ".txt", encoding = "utf8").read()

                #skip very short articles
                if len(articleDoc) > 50:
                    label = article.split("/")
                    label = label[len(label) - 2] + label[len(label) - 1]

                    articleLabels.append(label)
                    articleDocs.append(articleDoc)

    return articleDocs, articleLabels



#the java crawler missed these...
def filterArticles(articleTrainFolder):
    #get directories that contain the articles
    dirs = getArticleDirs(articleTrainFolder)
    print(len(dirs))

    i = 0
    for dir in dirs:
        for file in os.listdir(dir):
   
            if file.endswith(".txt"):
                article = dir + "/" + file
                articleDoc = open(article, encoding = "utf8").read()

                head, _, _ = articleDoc.partition("NOTA: Pentru a instaura un cadru civilizat de discuţii")
                head, _, _ = head.partition("_twitter_sess, auth_token, lang, twid, eu_cn, personalization_id,")
                
                #rewrite without these 
                if articleDoc != head:
                    with open(article, mode = 'w', encoding = "utf8") as repl:
                        repl.write(head)
                        
                        print(i)
                        i += 1

#get average relevant words for each publication
def getAverageSizePublication(articleTrainFolder):
    lemmaDict = initLematizer()
    stopwords = createStopwordsSet()

    dirs = getArticleDirs(articleTrainFolder)
    for dir in dirs:
        avg = 0
        totalArticles = 0
        for file in os.listdir(dir):   
            if file.endswith(".txt"):
                article = dir + "/" + file
                articleDoc = open(article, encoding = "utf8").read()

                totalArticles += 1
                avg += len(preprocessText(articleDoc, lemmaDict, stopwords))
        avg = avg/totalArticles
        print(dir + " " + str(avg))


#cleans a single directory
def dupFileremove(dir):
    duplicate = set()
    os.chdir(dir)
    path=os.getcwd()
    print ("The dir is: ", path)
    for filename in os.listdir(dir):
        filehash = None
        filepath=os.path.join(dir, filename)
        if os.path.isfile(filepath) and filepath.endswith('.txt') :
            filehash = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
        if filehash is not None and filehash not in duplicate:
            duplicate.add(filehash)
        elif filehash is not None:
            os.remove(filepath)
            os.remove(os.path.splitext(filepath)[0] + ".info")
            os.remove(os.path.splitext(filepath)[0] + ".tpcs")
            os.remove(os.path.splitext(filepath)[0] + ".ent")
            print("removed : ", filepath)

def cleanArticleFolder(articleTrainFolder):

    dirs = getArticleDirs(articleTrainFolder)
    i = 0
    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                article = dir + "/" + file
                articleDoc = open(article, encoding = "utf8").read()

                if (len(articleDoc) < 50):
                    #remove file
                    os.remove(dir + "/" + file)
                    #try to remove the other created files
                    try:
                        os.remove(os.path.splitext(dir + "/" + file)[0] + ".info")
                        os.remove(os.path.splitext(dir + "/" + file)[0] + ".tpcs")
                        os.remove(os.path.splitext(dir + "/" + file)[0] + ".ent")
                    except OSError:
                        print("OOPS ")
                        pass

        dupFileremove(dir)


def checkSequenceSimilarity(string1, string2):
    return SequenceMatcher(None, string1, string2).quick_ratio()

#filter function for a certain directory
#removes files that are very close to each other
#but might have been skipped by dupFileRemove
#WARNING: this is much slower, but filters
#WARNING: the results better
def filterDirectory(dir):
    i = 0
    cacheDict = {}
    for file in os.listdir(dir):
        if not file.endswith(".txt"):
            continue
        article = dir + "/" + file
        text = open(article, encoding="utf8").read().lower().split(" ")
        cacheDict[file] = text

    print("Finished caching the files")

    for file1 in os.listdir(dir):
        if file1.endswith(".txt"):
            for file2 in os.listdir(dir):
                if file2.endswith(".txt") and file1 != file2:
                    similarity = checkSequenceSimilarity(cacheDict[file1], cacheDict[file2])
                    if (similarity > 0.9):
                        os.remove(os.path.splitext(dir + "/" + file2)[0] + ".info")
                        os.remove(os.path.splitext(dir + "/" + file2)[0] + ".tpcs")
                        os.remove(os.path.splitext(dir + "/" + file2)[0] + ".ent")
                        os.remove(os.path.splitext(dir + "/" + file2)[0] + ".txt")
                        print(file1 + " " + file2 + " " + str(similarity))
                    if i % 10000 == 0:
                        print(i)
                    i +=1
                
#cache results so that we won't send too many requests
locationDict = {}
#converts names of locations to [lat, lon] pairs
#WARNING: this is very slow, consider not indexing geopoints
#         in elasticsearch
def getLocationsCoordinates(locations):
    from geopy import Nominatim
    from geopy.exc import GeocoderTimedOut
    import time

    geolocator = Nominatim()
    geoPoints = []

    print("Starting converting locations to coordinates")
    for loc in locations:
        if loc in locationDict:
            #check if not found before
            if locationDict[loc] != [-999, -999]:
                geoPoints.append(locationDict[loc])
        else:
            time.sleep(3)
            try:
                geoLoc = geolocator.geocode(loc)

                #continue if not found, but fill dictionary entry
                if not geoLoc:
                    locationDict[loc] = [-999, -999]
                    continue
                locationDict[loc] = [geoLoc.longitude, geoLoc.latitude]
                print(locationDict[loc])
                geoPoints.append(locationDict[loc])
            except GeocoderTimedOut as e:
                #limit the requests, sleep thread for a bit
                #in case Nominatim services denies our query
                print("Sleeping for 15 minutes")
                time.sleep(15 * 60)


    return geoPoints

#this is a different path from the path where the 
#java app saves the news (just to assure nothing goes wrong
#and we lose the data)
if __name__ == "__main__":
    #filterArticles("C:/textePublicatii/")
    #getAverageSizePublication("C:/textePublicatii/")
    #cleanArticleFolder("C:/textePublicatii/")
    #path = "C:/textePublicatii/www.realitatea.net/"
    #print(parseArticleTopicFile(path + "0", 10))
    filterDirectory("C:/textePublicatii/libertatea.ro")