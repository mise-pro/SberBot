import pandas as pd
import re
import numpy as np

import snowballstemmer
import nltk
import text_functions as tf
from sklearn.externals import joblib
import scipy


import string
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(mode):
    print('\n' + "loading data, please wait...")
    if mode == 'NORMAL':
        data_raw = pd.read_csv('vk.csv', sep=',', header=0, index_col=0)

        # удаляем ненужное
        questionsAnswers = data_raw.dropna(axis=0, how='any', inplace=False)

        questionsOriginal = questionsAnswers['question'].values
        answersOriginal = questionsAnswers['answer'].values

        # нормализация + очистка
        questionsNormalized = np.array([tf.clean_text(t) for t in questionsOriginal])

        # сохраняем все в дампы
        filename = 'DUMP_questionsAnswers.pkl'
        _ = joblib.dump(questionsAnswers, filename, compress=9)
        filename = 'DUMP_questionsOriginal.pkl'
        _ = joblib.dump(questionsOriginal, filename, compress=9)
        filename = 'DUMP_answersOriginal.pkl'
        _ = joblib.dump(answersOriginal, filename, compress=9)
        filename = 'DUMP_questionsNormalized.pkl'
        _ = joblib.dump(questionsNormalized, filename, compress=9)
    else:
        # загружаем дампы
        questionsAnswers = joblib.load('DUMP_questionsAnswers.pkl')
        questionsOriginal = joblib.load('DUMP_questionsOriginal.pkl')
        answersOriginal = joblib.load('DUMP_answersOriginal.pkl')
        questionsNormalized = joblib.load('DUMP_questionsNormalized.pkl')

    print('loading data DONE')
    return questionsAnswers, questionsOriginal, answersOriginal, questionsNormalized

def create_dictionary (questionsNormalized , lowLimit, maxLimit, mode):
    """
    Готовим словарь из нормированных вопросов
    """
    if mode == 'NORMAL':
        wordsAllNorm = []
        for question in questionsNormalized:
            for word in question:
                wordsAllNorm.append(word)
        print('Всего нормализованных слов:', len((wordsAllNorm)))
        wordsUniqNorm=set(wordsAllNorm)
        print('Всего уникальных нормализованных слов:',len((wordsUniqNorm)))

        wordsAllNormFreq = nltk.probability.FreqDist(wordsAllNorm)
        dictionaryWords = []
        for word in wordsAllNormFreq:
            if wordsAllNormFreq[word]>lowLimit and wordsAllNormFreq[word]<maxLimit:
                dictionaryWords.append(word)
        print('Отброшены слова повторяемостью меньше {} и больше {}'.format(lowLimit, maxLimit))

        # сохраняем в дамп
        filename = 'DUMP_dictionaryWords.pkl'
        _ = joblib.dump(dictionaryWords, filename, compress=9)
    else:
        # загружаем дампы
        print('<Загрузка из дампа>')
        dictionaryWords = joblib.load('DUMP_dictionaryWords.pkl')

    print('\n' + 'Всего слов в словаре:', len(dictionaryWords))
    return dictionaryWords

