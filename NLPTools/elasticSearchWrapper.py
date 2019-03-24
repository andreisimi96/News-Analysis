#Andrei Simion 2018

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Index, Text, DocType, Date, Keyword, GeoPoint, Search, Q
from elasticsearch_dsl.connections import connections
from articleWrapper import *
from dateutil import parser
from tfidf import tfidfWrapper
import gensim

tfidf_wrapper = tfidfWrapper("C:/textePublicatiiRelevante/")
tfidf_wrapper.trainTfidf()
#tfidf_wrapper.loadTfidfModel()

class Article(DocType):
    Title = Text(analyzer='snowball')
    Publication = Text()
    Link = Text()
    Content = Text(analyzer='snowball')
    Keywords = Keyword()
    Keywords_tfidf = Keyword()
    Publishing_Date = Date()
    #GeoLocations = GeoPoint()
    Locations = Text()
    Persons = Text()
    Organizations = Text()

    class Index:
        name = 'news'

    #call save method of the superclass to add article to elasticsearch
    def save(self, ** kwargs):
        return super(Article, self).save(** kwargs)

def deleteIndex(indexName):
    index = Index(indexName)
    index.delete(ignore=404)

def createIndex(indexName):
    index = Index(indexName)

    #register a doc_type with the index
    index.document(Article)
    index.create()

def addArticleToIndexDSL(articlePath):
    article = Article()

    document = open(articlePath + ".txt", encoding = "utf8").read()
    title, publishingDate, link, publication = parseArticleInfoFile(articlePath)
    persons, locations, organizations = parseArticleEntityFile(articlePath)
    keywords =  parseArticleTopicFile(articlePath, topKeywordsN=10)
    keywords_tfidf = tfidf_wrapper.getKeywordsForArticle(articlePath + ".txt")
    #print(keywords_tfidf)

    #initialize object with parsed data
    article.Title = title
    article.Publication = publication
    article.Link = link
    article.Content = document
    article.Keywords = keywords
    article.Keywords_tfidf = keywords_tfidf
    article.Publishing_Date = publishingDate
    article.Organizations = organizations
    article.Persons = persons
    #article.GeoLocations = getLocationsCoordinates(locations)
    article.locations = locations

    #label as used in doc2Vec
    articleLabel = articlePath.split('/')
    articleLabel = articleLabel[len(articleLabel) - 2] + articleLabel[len(articleLabel) - 1]    

    article.meta.id = articleLabel
    print('saving article ' + str(article.meta.id))

    article.save()

#returns article info based on articleID
#WARNING: articleID MUST match 
#WARNING: the doc label used in the doc2vec training
def searchArticle(articleID):

    s = Search()
    q = Q("multi_match", query= articleID, fields=['_id'])
    s = s.query(q)
    response = s.execute()

    #the id doesn't exist, return None
    if len(response.hits) == 0:
        return None

    hit = response.hits[0]

    title = hit.Title
    publication = hit.Publication
    publishing_date = parser.parse(hit.Publishing_Date)
    keywords = hit.Keywords
    
    return title, publication, publishing_date
    
#converts entire elasticsearch db to dict for caching
#should be quite speedy as it uses scan for bulk data extraction
#returns dict with keys as doc labels and values as tuples of the
#following form: (tile, publication, pubDate, Content)
def createESDict():

    #initialize connection to elasticsearch
    connections.create_connection(hosts=['localhost'])
    print(connections.get_connection().cluster.health())

    dict = {}
    
    #get standard search object
    s = Article.search()
    s.filter('range', _id= "")
    
    results = s.scan()
    i = 0
    for res in results:
        dict[res.meta.id] = (res.Title, res.Publication,
                             res.Publishing_Date, res.Content)

        i += 1
    return dict

def extractData(esDict, articleLabel):
    #write in plot relevant info, consider adding pub date as well?
    print(esDict[articleLabel][1])
    
    """wtf is this"""
    #publication = esDict[articleLabel][1].split('.')
    #publication = publication[1] + "." + publication[2] 
    publication = esDict[articleLabel][1]


    title = esDict[articleLabel][0]
    headline = publication + ": " + title[:40] + "..."

    text = esDict[articleLabel][3]
    return publication, title, headline, text

if __name__ == "__main__":

    #initialize connection
    connections.create_connection(hosts=['localhost'])
    print(connections.get_connection().cluster.health())

    
    """ WARNING: Decommenting this will cause the ENTIRE saved data 
                 within elasticsearch to be deleted"""
    deleteIndex('news')
    print('deleted index..')
    createIndex("news")

    dirs = getArticleDirs("C:/textePublicatiiRelevante/")
    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                article = dir + "/" + os.path.splitext(file)[0]
                addArticleToIndexDSL(article)