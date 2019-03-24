#Andrei Simion 2018

import time

#due to the limitations of polyglot
#on windows this module only supports python 2.7
#import polyglot
import os
from polyglot.text import Text, Word
from io import open

punctuation = [',', '.', '-']

#output entities to file as (entity.tag, entity appearances) pairs
def getEntitiesPG(file):
    fp = open(file, encoding="utf8")
    text = Text(fp.read(), hint_language_code="ro")

    output = os.path.splitext(file)[0] + ".ent"
    outfp = open(output, mode = 'w', encoding="utf8")

    #remembers number of occurences for each entity
    ent_dict = {}
    for entity in text.entities:
        string = entity.tag
        for token in entity:
            if token not in punctuation and token[0].isupper():
                string += " " + token

        if string != entity.tag:
            #check if it exists in dictionary
            if string in ent_dict:
                ent_dict[string] += 1
            else:
                ent_dict[string] = 1
    for entity in ent_dict:
        outfp.write(entity + " " + str(ent_dict[entity]) + "\n")

    outfp.close()

def writeEntitiesToDisk(articlesPath):
    dirs = [articlesPath + dir for dir in os.listdir(articlesPath)]
    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                article = dir + "/" + file
                getEntitiesPG(article)
        print("Finished dir: " + dir)

if __name__ == "__main__":
    articlesPath = "C:/textePublicatii/"
    writeEntitiesToDisk(articlesPath)