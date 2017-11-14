import collections
import sys
import re
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk.stem import PorterStemmer
from nltk.chunk import conlltags2tree, tree2conlltags


def jsonExtractor(filePath):
    # Method to extract the JSON file 
    # Input: Path to the JSON file
    # Output: Dictionary equivalent of JSON file

    source = open(filePath, 'r')
    print("SOURCE = ", source)
    jsonDataPoint = json.load(source)
    dictDataPoint = {}
    dictDataPoint = jsonDataPoint
    return dictDataPoint


def loadConfig(file):
    # Method to input values from config.json file
    # Input: Path to the config,json file
    # Output: Dictionary equivalent of config.json file
    config = jsonExtractor(file)
    return config


def loadStopWords(file):
    # Method to import list of stpo-words from a text file
    # Input: Path to the stop-word text file
    # Output: List of extracted stop-wpords
    stopWords = []
    loadStopWords = open(file, "r")
    stopWords = loadStopWords.read().split()
    return stopWords


def loadDocs(filePath):
    # Method to extract the 'Definition' and 'Description' from the given JSON file
    # Input: Path to the Json file
    # Output: Dictionary with Data Point(s) and corresponding 'Definition' and 'Description' 
    jsonTextDpDocs = jsonExtractor(filePath)
    mappedValues = sentenceMapper(jsonTextDpDocs)
    return mappedValues


def serialize(obj, file):
    # Method to serialize and pickle an object
    # Input: Object to be pickled, Destination of created pickle file
    # Output: Pickle fle at the designated path
    import pickle
    filehandler = open(file, "wb")
    pickle.dump(obj, filehandler)
    filehandler.close()


def deserialize(file):
    # Method to de-serialize and un-pickle a pickled object
    # Input: Path to the Pickled Object
    # Output: Unpickled object
    import pickle
    print("FILE = ", file)
    filehandler = open(file, "rb")
    print("DESERIALIZE FILEHANDLER = ", filehandler)
    obj = pickle.load(filehandler)
    filehandler.close()
    return obj


def regexExtractor(text):
    # Method to remove non-alphabetical characters and spaces
    # Input: Text to be cleaned using Regular Expression
    # Output: Cleaned Text
    NewText = re.sub('\s+',' ', text)
    notAlphanumericRegex = re.compile(r'[^A-Za-z]')
    excludedNotAlphanumeric = notAlphanumericRegex.findall(NewText)
    postNotAlphanumericRegex = re.sub(notAlphanumericRegex, ' ', NewText)
    cleansedText = postNotAlphanumericRegex
    return cleansedText

lemmatizer = WordNetLemmatizer()
def lemmatizeWord(word, posTag):
    # Method for Lemmatization
    # Input: word to be lemmatized, corresponding POS Tag of the word
    # Output: lemmatized word, POS Tag    
    return lemmatizer.lemmatize(word, pos=posTag)

def regexExtractorExt(text):
    # Method to remove numbers and special characters
    # Input: Text to be cleaned using Regular Expression
    # Output: Cleaned Text
    cleansedText = []
    numeralRegex = re.compile(r' [0-9]+ ')
    postNumeralRegex = re.sub(numeralRegex, ' ', text)
    notAlphanumericRegex = re.compile(r'[^A-Z0-9a-z-\. ]')
    postNotAlphanumericRegex = re.sub(notAlphanumericRegex, ' ', postNumeralRegex)
    seeMissingRegex = re.compile(r' -')
    postSeeMissingRegex = re.sub(seeMissingRegex, ' ', postNotAlphanumericRegex)
    sawMissingRegex = re.compile(r'- ')
    postSawMissingRegex = re.sub(sawMissingRegex, ' ', postSeeMissingRegex)
    cleansedText.append(re.sub('\s+', ' ', postSawMissingRegex))
    cleanRegex = ' '.join(cleansedText)
    return cleanRegex


def namedEntityRecognition(pos):
    chunkedToken = ne_chunk(pos)
    namedEntity = tree2conlltags(chunkedToken)
    return namedEntity


def preProcessVocab(clean_text,docId, stpWords):
    lemmatizer = WordNetLemmatizer()
    nounList = ["NN", "NNS"]
    adverbList = ["RB", "RBR", "RBS"]
    adjectiveList = ["JJ", "JJR", "JJS"]
    verbList = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    stemmer = PorterStemmer()
    
    dpVocab = []

    sentences = nltk.sent_tokenize(clean_text)
    j = 0
    for sentence in sentences:
        # sentenceNew = splitString(sentence)
        tokenList = nltk.word_tokenize(sentence)
        partOfSpeech = nltk.pos_tag(tokenList)
        neChunkedToken = namedEntityRecognition(partOfSpeech)
        k = 0
        for token in tokenList:
            if token.lower() in stpWords and re.search(r'no+', token.lower()) is None:
                k += 1
                continue
            isNamedEntity = False
            stemValue = stemmer.stem(token)
            posValue = neChunkedToken[k][1]

            if posValue in nounList:
                lemmaValue = lemmatizeWord(token, 'n')
            elif posValue in adverbList:
                lemmaValue = lemmatizeWord(token, 'r')
            elif posValue in adjectiveList:
                lemmaValue = lemmatizeWord(token, 'a')
            elif posValue in verbList:
                lemmaValue = lemmatizeWord(token, 'v')
            else:
                lemmaValue = lemmatizer.lemmatize(token)
            if posValue == 'NNP' or posValue == 'NNPS':
                isNamedEntity = True

            # Creating Vocabulary Entry Object. We can further reduce the JSON
            # Output by removing repeating JSON Objects, which is not
            # implemented yet.
            dpVocab.append(IndexEntry(token,lemmaValue,docId))
            k += 1
        j += 1
    return dpVocab


def sentenceMapper(jsonObject):
    # Method to extract the 'Definition' and 'Description' from the given JSON file
    # Input: Path to the Json file
    # Output: Dictionary with Data Point(s) and corresponding 'Definition' and 'Description' 
    mappedSentences = {}
    for obj in jsonObject:
        sentenceList = []
        definitionValue = obj.get("definition")
        descriptionValue = obj.get("description")
        idValue = obj.get("id")
        if definitionValue is not None:
            sentenceDefinition = nltk.sent_tokenize(definitionValue)
        else:
            sentenceDefinition = nltk.sent_tokenize("This is a dummy sentence.")
        if descriptionValue is not None:
            sentenceDescription = nltk.sent_tokenize(descriptionValue)
        else:
            sentenceDescription = nltk.sent_tokenize("This is a dummy sentence.")

        sentenceList.append(sentenceDefinition)
        sentenceList.append(sentenceDescription)
        mappedSentences[idValue] = sentenceList
    return mappedSentences


def createAnalyzer(analyzerOptions):
    # Method to create an Analyzer object based on the Analyzer configurations in the config.json file
    # Input: Analyzer configuration block from config.json
    # Output: Corresponding Analyzer object
    if (analyzerOptions["type"] == "analyzer-1"):
        print("Analyzer-1")
        stopWords = loadStopWords(analyzerOptions["stop-words-file"])
        return Analyzer1(stopWords)
    elif (analyzerOptions["type"] == "lemma"):
        stopWords = loadStopWords(analyzerOptions["stop-words-file"])
        return NaiveLemmaAnalyzer(stopWords)
    elif (analyzerOptions["type"] == "surface"):
        return NaiveSurfaceAnalyzer()


def getTopMatches(matches, n):
    # Method to return top 'n' matches from a list
    # Input: List of Key-value pairs, n
    # Output: Sorted list of Key-value pairs 
    sortedMatches = sorted(matches, key=lambda tup: tup[1], reverse=True)
    return sortedMatches[:n]


def createEngine(engineOptions):
    # Method to create an Engine object based on the Engine configurations in the config.json file
    # Input: Engine configuration block from config.json
    # Output: Corresponding Engine object
    index = deserialize(engineOptions["index-file"])
    if (engineOptions["type"] == "boolean"):
        return BooleanEngine(index)
    elif (engineOptions["type"] == "tf-idf"):
        return TfIdfEngine(index, engineOptions["n"])
    elif (engineOptions["type"] == "bm25"):
        return Bm25Engine(index, engineOptions["n"], engineOptions["k"], engineOptions["b"])
    else:
        raise ValueError("bad engine option")


def runInputSearchPrintLoop(engine, analyzer):
    # TODO: a while loop waiting for input, terminates the UI on say ctrl+D
    # in the loop:
    

    termToMatch=[]
    resultDict = {}

    inputStr = input("\n\t\tEnter search keywords here.  Or Enter 'exit' to terminate: ")
    if inputStr == "EXIT" or inputStr == "exit":
        sys.exit(0)
    terms = analyzer.analyze("DUMMY", inputStr)
    for term in terms:
        termToMatch.append(term.getForm().lower())

    # normalized = []
    # for token in termToMatch:
    #     token = token.replace(token[0], token[0].upper())
    # normalized.append(token)

    # termToMatch.extend(normalized)

    matches = engine.getMatches(termToMatch)
    resultDict["input"] = inputStr
    resultDict["matches"] = matches
    return resultDict


class Analyzer1(object):
    # analyzer with one stop-word list - just making stuff up for illustration

    def __init__(self, stopWords):
        self.stopWords = stopWords

    def analyze(self, docId, text):
        regexClean = regexExtractor(str(text))
        tokenizedQueryList = preProcessVocab(regexClean,docId, self.stopWords)
        return tokenizedQueryList


class NaiveSurfaceAnalyzer(object):
    # analyzer with no stop words - just making stuff up for illustration

    def __init__(self):
        pass

    def lemmatizeWord(self, word, posValue):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        mapping = {('NN', 'NNS') : 'n', ("RB", "RBR", "RBS") : 'r', ("JJ", "JJR", "JJS") : 'a', ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ") : 'v'}
        posTag = mapping.get(posValue, word)
        if posTag is not word:
            lemmaValue = lemmatizer.lemmatize(word, posTag)
        else:
            lemmaValue = lemmatizer.lemmatize(word)

        return lemmaValue

    def analyze(self, dataPointName, text):
        # TDOD: text is a string 
        # apply the character filter, tokenization, token filter, etc.
        # return a list of generated tokens

        parsedOutput = regexExtractor(str(text))
        tokens = nltk.word_tokenize(parsedOutput.lower())
        posTags = nltk.pos_tag(tokens)

        listOfObjects = []
        for values in posTags:
            lemma = self.lemmatizeWord(values[0], values[1])
            
            #indexEntryObject = IndexEntry(lemma, values[1], dataPointName, text)
            indexEntryObject = IndexEntry(values[0], values[1], dataPointName)
            listOfObjects.append(indexEntryObject)
        return listOfObjects


class NaiveLemmaAnalyzer(object):
    # analyzer with no stop words - just making stuff up for illustration

    def __init__(self, stopWords):
        self.stopWords = stopWords

    def analyze(self, dataPointName, text):
        # TDOD: text is a string 
        # apply the character filter, tokenization, token filter, etc.
        # return a list of generated tokens
        
        parsedOutput = regexExtractor(str(text))
        tokens = nltk.word_tokenize(parsedOutput.lower())
        posTags = nltk.pos_tag(tokens)

        listOfObjects = []
        k=0
        for values in posTags:
            nounList = ["NN", "NNS"]
            adverbList = ["RB", "RBR", "RBS"]
            adjectiveList = ["JJ", "JJR", "JJS"]
            verbList = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

            if values[0].lower() in self.stopWords:
                k += 1
                continue
            
            if values[1] in nounList:
                lemma = lemmatizeWord(values[0], 'n')
            elif values[1] in adverbList:
                lemma = lemmatizeWord(values[0], 'r')
            elif values[1] in adjectiveList:
                lemma = lemmatizeWord(values[0], 'a')
            elif values[1] in verbList:
                lemma = lemmatizeWord(values[0], 'v')
            else:
                lemma = lemmatizer.lemmatize(values[0])
            
            indexEntryObject = IndexEntry(lemma, values[1], dataPointName)
            # indexEntryObject = IndexEntry(values[0], values[1], dataPointName)
            listOfObjects.append(indexEntryObject)
        return listOfObjects


class IndexEntry(object):
    def __init__(self, form, value, dataPoint):
        self.form = form
        self.value = value
        self.dataPoint = dataPoint
        

    def getForm(self):
        #print("FORM = ", self.form)
        return self.form

    def getValue(self):
        return self.value

    def getDataPoint(self):
        return self.dataPoint


class Result(object):
    def __init__(self, idValue, name, definition, description):
        self.idValue = idValue
        self.name = name
        self.definition = definition
        self.description = description

        

    def getIdValue(self):
        #print("FORM = ", self.form)
        return self.idValue

    def getName(self):
        return self.name

    def getDefinition(self):
        return self.definition

    def getDescription(self):
        return self.description



class Index(object):

    def __init__(self, indexDict, docSizeDict, avrgD):
        self.indexDict = indexDict
        self.docSizeDict = docSizeDict
        self.avrgD = avrgD

    def getAvrgD(self):
        return self.avrgD

    def getD(self, docId):
        return self.docSizeDict[docId]

    def hasTerm(self, term):
        if term in self.indexDict.keys():
            return True
        else:
            return False

    def getIdf(self, term):
        idf = 0.0
        if (self.hasTerm(term)):
            idf = self.indexDict[term]["idf"]
        return idf

    def getEntries(self, term):
        entries = []
        # print(self.docSizeDict)
        if (self.hasTerm(term)):
            entries = self.indexDict[term]["data"]
        return entries


class BooleanEngine(object):

    def __init__(self, indexObj):
        self.index = indexObj

    def getMatches(self, terms):
        '''
        returns a list of matching pairs (doc-id, score)
        '''
        docIds = {entry["doc-id"] for entry in self.index.getEntries(terms[0])}
        print()
        for t in terms[1:]:
            d = {entry["doc-id"] for entry in self.index.getEntries(t)}
            docIds = docIds & d  # intersection is for AND-search [union would be for OR-search]
        return [(docId, 1.0) for docId in docIds]  # add a constant score of 1.0 to all, for consistency wit the other engines


class TfIdfEngine(object):

    def __init__(self, index, n):
        self.index = index
        self.n = n

    def getMatches(self, terms):
        '''
        returns a list of top-n matching pairs (doc-id, score)
        ranked by descending scores
        '''
        matches = collections.defaultdict(float)
        for t in terms:
            if (self.index.hasTerm(t) is None):
                continue
            idf = self.index.getIdf(t)
            for entry in self.index.getEntries(t):
                docId = entry["doc-id"]
                tf = entry["tf"]
                matches[docId] += tf * idf
        return getTopMatches([(k, v) for k, v in matches.items()], self.n)


class Bm25Engine(object):

    def __init__(self, index, n, k, b):
        self.index = index
        self.n = n
        self.k = k
        self.b = b
        self.avrgD = index.getAvrgD()

    def getMatches(self, terms):
        '''
        returns a list of top-n matching pairs (doc-id, score)
        ranked by descending scores
        '''
        matches = collections.defaultdict(float)
        for t in terms:
            if (self.index.hasTerm(t) is None):
                continue
            idf = self.index.getIdf(t)
            for entry in self.index.getEntries(t):
                docId = entry["doc-id"]
                tf = entry["tf"]
                d = self.index.getD(docId)
                w = (idf * tf * (1 + self.k)) / (tf + self.k * (1 - self.b + self.b * d / self.avrgD))
                matches[docId] += w
        return getTopMatches([(k, v) for k, v in matches.items()], self.n)


def index(config):
    analyzer = createAnalyzer(config["analyzer-options"])
    mappedSentences = loadDocs(config["docs-file"])  # TODO implement loadDocs()

    listOfValues = []
    for dataPoint in mappedSentences.items():
        for sentence in dataPoint[1]:
            for entry in analyzer.analyze(dataPoint[0], sentence):
                listOfValues.append(entry)


    indexDict = {}  # TODO
    docSizeDict = {}  # TODO

    for obj in listOfValues:

        if obj.getDataPoint() not in docSizeDict:
            docSizeDict[obj.getDataPoint()] = 1
        else:
            docSizeDict[obj.getDataPoint()] += 1

        indexTerm = obj.getForm().lower()
        if indexTerm not in indexDict:
            indexDict[indexTerm] = {}
            data = []
            inFileOccurences = {}
            inFileOccurences["doc-id"] = obj.getDataPoint()
            inFileOccurences["tf"] = 1
            data.append(inFileOccurences)
            indexDict[indexTerm] = {}
            indexDict[indexTerm]["data"] = data
        else:
            for row in indexDict.items():
                if row[0] == indexTerm:
                    foundFlag = False
                    for docs in row[1]["data"]:
                        if docs["doc-id"] == obj.getDataPoint():
                            docs["tf"] += 1
                            foundFlag = True
                    if foundFlag is False:
                        newFileOccurences = {}
                        newFileOccurences["doc-id"] = obj.getDataPoint()
                        newFileOccurences["tf"] = 1
                        row[1]["data"].append(newFileOccurences)

    totalDocsSize = len(docSizeDict)
    modifiedIndexDict = indexDict.copy()
    avrgD = len(listOfValues) / totalDocsSize

    import math
    for entry in indexDict.items():
        for row in entry[1]["data"]:
            docID = row["doc-id"]
            locatedInFileCount = len(entry[1]["data"])
            word = entry[0]
            computedValue = totalDocsSize / (locatedInFileCount + 1)
            idfValue = math.log(computedValue, 10)
        modifiedIndexDict[word]["idf"] = idfValue
    index = Index(indexDict, docSizeDict, avrgD)
    outputIndexPath = config["engine-options"]["index-file"]
    print("OUTPUT = ", outputIndexPath)
    newIdexedJsonText = json.dumps(indexDict, indent = 4)
    with open(config["engine-options"]["index-json"], 'w') as outfile:
        outfile.write(newIdexedJsonText)
    serialize(index, outputIndexPath)


def search(config):
    analyzer = createAnalyzer(config["analyzer-options"])
    engine = createEngine(config["engine-options"])

    while(True):
        matchList = runInputSearchPrintLoop(engine, analyzer)
        resultList = []
        jsonTextDpDocs = jsonExtractor(config["docs-file"])

        for match in matchList["matches"]:
            for obj in jsonTextDpDocs:
                if match[0] == obj.get("id"):
                    description = obj.get("description")
                    definition = obj.get("definition")
                    name = obj.get("name")
                    resultList.append(Result(match[0], name, description, definition)) 

        print("\nTotal Results returned: " + str(len(resultList)))
        for obj in resultList:
            print("\nDOC = ", obj.getName(), "\nVALUE = ", obj.getIdValue())
            print("DESCRIPTION = ", obj.getDescription())
            print("DEFINITION = ", obj.getDefinition())

        with open(config["engine-options"]["output-path"], 'w') as outfile:
            header = str(config["engine-options"]["type"]) + " " + str(config["analyzer-options"]["type"]) + " RESULTS for " + str(matchList["input"]) + ":"
            outfile.write(str(header))
            outfile.write("\nTotal Results returned: " + str(len(resultList)))
            for obj in resultList:
                doc = obj.getName()
                value = obj.getIdValue()
                descriptionValue = obj.getDescription()
                definitionValue = obj.getDefinition()
                outfile.write("\n\nDOC: " + str(doc))
                outfile.write("\nValue: " + str(value))
                outfile.write("\nDescription: " + str(descriptionValue))
                outfile.write("\nDefinition: " + str(definitionValue))


if __name__ == "__main__":
    doWhat = sys.argv[1]
    config = loadConfig(sys.argv[2])
    print("CONFIG = ", config)
    if (doWhat == "search"):
        search(config)
    elif (doWhat == "index"):
        index(config)
    else:
        raise ValueError("bad doWhat option")
