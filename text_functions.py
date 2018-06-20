import re
import numpy as np

import snowballstemmer
import nltk

import string

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = snowballstemmer.RussianStemmer()

#todo необходимо улучшить качество предпроцессинга
def text_preprocessing(text):
    text = text.lower() # приведение в lowercase,
    
    text = re.sub( r'https?://[\S]+', ' url ', text) # замена интернет ссылок
    text = re.sub( r'[\w\./]+\.[a-z]+', ' url ', text) 
 
    text = re.sub( r'\d+[-/\.]\d+[-/\.]\d+', ' date ', text) # замена даты и времени
    text = re.sub( r'\d+ ?гг?', ' date ', text) 
    text = re.sub( r'\d+:\d+(:\d+)?', ' time ', text) 

    # text = re.sub( r'@\w+', ' tname ', text ) # замена имён twiter
    # text = re.sub( r'#\w+', ' htag ', text ) # замена хештегов

    text = re.sub( r'<[^>]*>', ' ', text) # удаление html тагов
    text = re.sub( r'[\W]+', ' ', text ) # удаление лишних символов

   
    text = re.sub( r'\b\w\b', ' ', text ) # удаление отдельно стоящих букв

    text = re.sub( r'\b\d+\b', ' digit ', text ) # замена цифр 

    #return text
    return re.sub(r'\s+', ' ', text)

	#http://zabaykin.ru/?tag=nltk
def tokenize_text(fileText):
    fileText=text_preprocessing(fileText)
    
    #firstly let's apply nltk tokenization
    tokens = nltk.word_tokenize(fileText)

    #let's delete punctuation symbols
    tokens = [i for i in tokens if ( i not in string.punctuation )]

    #cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]
    return tokens
	
	
def clean_text(text):
    tokens = tokenize_text(text)
    tokens = [token for token in tokens if len(token) > 3]
    
    #deleting stop_words
    #todo надо использовать 1 объект
    stopWords = nltk.corpus.stopwords.words('russian')
    stopWords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на','ею','котор','месяц'])
    tokens = [i for i in tokens if (i not in stopWords )]
    #stemmer = snowballstemmer.RussianStemmer()
    tokens = [stemmer.stemWord(token) for token in tokens]
    
    tokens = [i for i in tokens if (i not in ['подскаж','digit','date', 'спасиб','пожалуйст'] )]
    
    tokens = [token for token in tokens if len(token) > 3]
    
    #delete containing first digits
    tokens = [token for token in tokens if not token[0].isdigit()]
    return tokens

	
def get_top_freq_words(questionsNormalized, topPositions=10, freqCount=True):
    """
    считает топы слов первых эн штук
    """
    wordsAllNorm=[]
    for question in questionsNormalized:
        for word in question:
            wordsAllNorm.append(word)
    wordsAllNormFreq = nltk.probability.FreqDist(wordsAllNorm)
    if freqCount:
        return wordsAllNormFreq
    elif topPositions is None:
        return sorted(wordsAllNormFreq, key=wordsAllNormFreq.get, reverse=True)
    else:
        return sorted(wordsAllNormFreq, key=wordsAllNormFreq.get, reverse=True)[:topPositions]

def show_top_freq_words(questionsNormalized, topPositions):
    """
    вывоидт топы слов и выводит первых эн штук
    """
    wordsAllNormFreq = get_top_freq_words(questionsNormalized, topPositions, freqCount=True)
    for w in sorted(wordsAllNormFreq, key=wordsAllNormFreq.get, reverse=True)[:topPositions]:
        print (w, wordsAllNormFreq[w])
    return