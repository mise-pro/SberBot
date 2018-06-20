# coding=utf-8
from sklearn.externals import joblib
#from sklearn.datasets import load_digits
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import text_functions as tf
from sklearn.model_selection import GridSearchCV

def load_best_lda(calculatedModel, calculatedModelOutput):
    print('\n' + 'Загрузка модели...')
    ldaModels = joblib.load(calculatedModel)

    # Best Model
    bestLdaModel = ldaModels.best_estimator_
    ldaOutput = joblib.load(calculatedModelOutput)

    print('Загрузка модели завершена!')
    return bestLdaModel, ldaOutput


def grid_search_best_lda(searchParams, calculatedModels, dataVectorized):
    if not calculatedModels:
        # Init the Model
        print('\n' + 'Поиск лучшей модели будет осуществляться заново, ждите долго ...')
        lda = LatentDirichletAllocation(n_jobs=-1)
        # Init Grid Search Class
        model = GridSearchCV(lda, param_grid=searchParams)
        # Do the Grid Search
        gsmodels = model.fit(dataVectorized)
        # saveToFile
        filename = 'DUMP_LDAModels.pkl'
        _ = joblib.dump(gsmodels, filename, compress=9)

        # Best Model
        bestLdaModel = gsmodels.best_estimator_
        filename = 'DUMP_bestLDAModel.pkl'
        _ = joblib.dump(gsmodels, filename, compress=9)
    else:
        print('\n' + 'Загрузка модели из кеша')
        filename = 'DUMP_LDAModels.pkl'
        gsmodels = joblib.load(filename)

        filename = 'DUMP_bestLDAModel.pkl'
        bestLdaModel = joblib.load(filename)


    if not calculatedModels:
        ldaOutput = bestLdaModel.transform(dataVectorized)
        # saveToFile
        filename = 'DUMP_bestLDAModelOutput.pkl'
        _ = joblib.dump(ldaOutput, filename, compress=9)
    else:
        filename = 'DUMP_bestLDAModelOutput.pkl'
        ldaOutput = joblib.load(filename)

    # Model Parameters
    print("Best Model's Params: ", gsmodels.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score from all models checked: ", gsmodels.best_score_)

    # Get Log Likelyhoods from Grid Search Output
    n_topics = searchParams['n_components']
    log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in gsmodels.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.5]
    log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in gsmodels.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.7]
    log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in gsmodels.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()
    return bestLdaModel, ldaOutput


def get_num_topic(questionVect, ldaModel):
    topicProbabilityScores = ldaModel.transform(questionVect)
    return np.argmax(topicProbabilityScores)


def get_5_nearest_Q_and_A_via_cos(currentQuestionVectorized, questionsVectorized, isDebug=False):
    # функция поиска 5 ближайших расстояний через косинус
    top5MinDist = np.zeros(5)
    top5MinDistIdx = np.zeros(5, dtype=int)

    for idx in range(questionsVectorized.shape[0]):
        if (calculate_cos_dist(currentQuestionVectorized, questionsVectorized[idx])) > (np.amin(top5MinDist)):
            top5MinDistIdx[np.argmin(top5MinDist)] = idx
            top5MinDist[np.argmin(top5MinDist)] = calculate_cos_dist(currentQuestionVectorized,
                                                                           questionsVectorized[idx])
    if isDebug:
        print(top5MinDist)
        print(top5MinDistIdx)

    return top5MinDist, top5MinDistIdx


def show_5_nearest_Q_A_prob(questions, answers, top5Probs, top5MinDistIdx, isDebug=False):
    # функиця показа 5 ближайших расстояний
    for idx in top5MinDistIdx:
        print('ВОПРОС:', questions[idx])
        print('ОТВЕТ:', answers[idx])
        print('ВЕРОЯТНОСТЬ:', top5Probs[np.where(top5MinDistIdx == idx)[0][0]])
        if isDebug:
            print('НОМЕР ВОПРОСА В МАССИВЕ:', idx)
        print()
    return


def get_5_nearest_Q_and_A_via_lda_probs(probs, isDebug=False):
    # функиця поиска 5 ближайших расстояний через вероятность модели
    probs = np.array(probs)
    top5MinDist = []
    top5MinDistIdx = np.argsort(-probs)[:5]

    for idx in top5MinDistIdx:
        top5MinDist.append(probs[idx])
    if isDebug:
        print(top5MinDist)
        print(top5MinDistIdx)

    return top5MinDist, top5MinDistIdx


def calculate_cos_dist(u, v):
    return np.dot(u, v) / norm(u) / norm(v)  # cosine of the angle


def calculate_summary_stats_for_method(currentQuestionVectorized, questionVect):
    # функиця для расчета очки вероятности по 5ти ближайшим вопросам для кос/не кос
    result = .0
    for currentQuestionVectorizedIter in questionVect:
        result += calculate_cos_dist(currentQuestionVectorized, currentQuestionVectorizedIter)
    print("Сводное значение, рассчитаное через cos= ", round(result, 5))


def show_most_freq_words_per_theme_via_lda(ldaModel, allWords, nWords=15, isDebug=False):
    # показать 15 ключевых слов из каждой темы через короткие нормированные слова
    print('\n' + 'Все ключевые нормированые слова в найденных темах:')
    wordsPerTheme = get_most_freq_words_per_theme_via_lda(ldaModel, allWords, nWords)

    themeNum = 0
    for words in wordsPerTheme:
        print('Тема {}:'.format(themeNum), words)
        themeNum += 1


def get_most_freq_words_per_theme_via_lda(ldaModel, allWords, nWords=15):
    # получить 15 ключевых слов из каждой темы через короткие нормированные слова

    allWords = np.array(allWords)
    result = []
    for wordsPerTheme in ldaModel.components_:
        topKeywordsIdx = np.argsort(-wordsPerTheme)[:nWords]
        result.append(allWords[topKeywordsIdx])
    return result


def show_most_freq_words_per_theme(mostFreqWordsPerThemes):
    # Все ключевые ненормированые слова в найденных темах
    print('\n' + 'Все ключевые слова в темах модели:')
    themeNum = 0
    for fullWords in mostFreqWordsPerThemes:
        print('Тема {}: {}'.format(themeNum, ' '.join(fullWords)))
        themeNum += 1


def get_most_freq_words_per_theme(allWords, ldaModel, questionsOriginal):
    # функция, которая для ключевых коротких слов темы
    # подберет их аналог из наиболее часто встречающихся в оригинальных вопросах

    # находим уникальные слова в темах, которые получились
    allWords = np.array(allWords)
    TWordsUniq = []
    for wordsPerTheme in ldaModel.components_:
        topKeywordsIdx = np.argsort(-wordsPerTheme)[:15]
        TWordsUniq.extend(allWords[topKeywordsIdx])
    TWordsUniq = set(TWordsUniq)

    # найдем самые частые слова в оригинальных вопросах
    questionsOriginalTokenized = np.array([tf.tokenize_text(t) for t in questionsOriginal])
    allWordsSortedDesc = tf.get_top_freq_words(questionsOriginalTokenized, None, freqCount=False)

    # и оставим там только уникальные (чтобы ниже потом меньше циклов делать по длинным циклам)
    fullVersionOfWordsList = set()
    for TWordUniq in TWordsUniq:
        for word in allWordsSortedDesc:
            if TWordUniq in word:
                fullVersionOfWordsList.add(word)
                break

    mostFreqWordsPerThemes = get_most_freq_words_per_theme_via_lda(ldaModel, allWords)

    # заменим в mostFreqWordsPerThemes слова на значения из fullVersionOfWordsList
    for wordsOfTheme in mostFreqWordsPerThemes:
        for wordIdx in range(len(wordsOfTheme)):
            for fullWord in fullVersionOfWordsList:
                if (wordsOfTheme[wordIdx]) in fullWord:
                    wordsOfTheme[wordIdx] = fullWord
                    break
    return mostFreqWordsPerThemes
