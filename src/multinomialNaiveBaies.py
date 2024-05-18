import collections
import math 

class MultinomialNB:
    def __init__(self, articles_per_tag):
        # Don't change the following two lines of code.
        self.articles_per_tag = articles_per_tag  # See question prompt for details.
        self.train()

    def getProperties(self):
        
        self.articlesCounter = {}
        for key, value in self.articles_per_tag.items():
            self.articlesCounter.update(
                {
                    key: len(value)
                }
            )

        self.totArticles = sum(self.articlesCounter.values())
        
        # computing prior
        self.tagPrior = self.articlesCounter.copy()
        for key, value in self.tagPrior.items():
            self.tagPrior[key] = value / self.totArticles
        
        print('totArticles  = ', self.totArticles)
        print('tagPrior     = ', self.tagPrior, '\n')
        
    def vocabularyGenerator(self):
        vocabulary = collections.Counter()
        for _, articlesList in self.articles_per_tag.items():
            for article in articlesList:
                vocabulary = vocabulary + collections.Counter(article)

        self.vocabulary = vocabulary

        print('vocabulary entries = ', len(self.vocabulary.keys()), '\n')

    def likelyhoodGenerator(self):
        
        self.likelyhood = {}
        
        for tag, articleList in self.articles_per_tag.items():
            
            articlesWords = collections.Counter()
            
            for article in articleList:
                articlesWords = articlesWords + collections.Counter(article)

            totalWords = sum(articlesWords.values()) 
            print(tag, 'totalWords = ', totalWords)

            likelyhoodDict = self.vocabulary.copy()
            
            for word in self.vocabulary.keys():
                if word in articlesWords.keys():
                    wordCounter = articlesWords[word]
                else:
                    wordCounter = 0.0

                # likelyhood with lagrange smoothing
                likelyhood = (wordCounter + 1) / (totalWords + 2)

                likelyhoodDict[word] = likelyhood

            tagDict = {
                tag: likelyhoodDict
            }
            
            # print(tagDict)            
            
            self.likelyhood.update(tagDict)

    def vectorizing(self, article):
        
        articlesWords = collections.Counter(article)

        vector = self.vocabulary.copy()
        
        for word, _ in self.vocabulary.items():
            if word in articlesWords.keys():
                wordCounter = articlesWords[word]
            else:
                wordCounter = 0.0

            vector[word] = wordCounter 

        # print('vector = ', vector)

        return vector 
    
    def train(self):
        self.getProperties()
        self.vocabularyGenerator()
        self.likelyhoodGenerator()

    def predict(self, article):
        prediction = {}

        vector = collections.Counter(article)
        
        # vector = self.vectorizing(article)
        
        for tag in self.articles_per_tag.keys():
            pred_ = 0
            
            for word in vector.keys():
                if word in self.vocabulary.keys():
                    pred_ = pred_ + vector[word] * math.log(self.likelyhood[tag][word])           
                else:
                    pred_ = pred_ + vector[word] * math.log(0.5)
                    
            pred_ = math.log(self.tagPrior[tag]) + pred_
            prediction.update(
                {
                    tag: pred_
                }
            )

        return prediction
            