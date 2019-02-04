from __future__ import unicode_literals
import string
import pandas as pd
import codecs
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
pd.set_option('display.width', 2000)

class PreProcessing:
    Data = None
    Language = None
    def __init__(self,Data,language):
        self.Data = Data
        self.Language = language

    def Remove_Punctuation(self):
        # if language == 'persian':
        if self.Language == 'persian':
            count = 1
            # print (" Removing punctuation...")
            for i in range(len(self.Data)):
                # print (count)
                count += 1
                try:
                    mess = self.Data.loc[i, 'Text']
                    nopunc = [char for char in mess if char not in string.punctuation]
                    nopunc = ''.join(nopunc)
                    self.Data.loc[i, 'Text'] = nopunc
                except:
                    pass
            return self.Data
        else:
            # print("Removing punctuation...")
            for i in range(len(self.Data)):
                # print (count)
                try:
                    mess = self.Data[i]
                    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
                    self.Data[i] = mess.translate(translator)
                except:
                    # print('in except')
                    pass
            return self.Data

    def Remove_StopWords(self):
        stopwords_persian = '../stopwords-persian'
        # stopwords_english = '../stopwords-english'
        stopwords_list = set(stopwords.words(self.Language))
        if self.Language == 'persian':
            f = codecs.open(stopwords_persian, encoding='utf-8')
            stopwords_persian = f.read()
            count = 1
            # print ("Removing StopWords...")
            for i in range(len(self.Data)):
                # print(count)
                count += 1
                try:
                    text = (self.Data.loc[i, 'Text']).split()
                    filtered = [word for word in text if word not in stopwords_persian.split()]
                    filtered = ' '.join(filtered)
                    self.Data.loc[i, 'Text'] = filtered
                except:
                    pass
        #if self.Language == 'english':
        # print ("Removing StopWords...")
        for i in range(len(self.Data)):
            try:
                words = word_tokenize(self.Data[i])
                wordsFiltered = [word for word in words if word not in stopwords_list]
                wordsFiltered = ' '.join(wordsFiltered)
                self.Data[i] = wordsFiltered
            except:
                pass
        return self.Data
