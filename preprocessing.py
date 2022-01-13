from nltk import stem, word_tokenize
from nltk.corpus import stopwords
import re
import copy
import pandas as pd

from aspect_matcher import AspectMatcher


# --Unify the notation of sentiment to POS, NEG and NEU
def sentiment_unification(data):
    pos_list = ['positive', '1', 'Positive', 'positve', 'positive ', 'pos']
    neg_list = ['negative', '-1', 'Negative', 'neg']
    neu_list = ['0', 'negative/positive', 'discarded', 'neu']
    for index in range(len(data.sentiment)):
        for index2 in range(len(pos_list)):
            if pos_list[index2] == data.sentiment[index]:
                data.sentiment[index] = 'POS'
                break
        for index2 in range(len(neg_list)):
            if neg_list[index2] == data.sentiment[index]:
                data.sentiment[index] = 'NEG'
                break
        for index2 in range(len(neu_list)):
            if neu_list[index2] == data.sentiment[index]:
                data.sentiment[index] = 'NEU'
                break
    return data


def padding_sentiment(data):
    for index in range(len(data["sentiment"])):
        aspects = data["aspect"][index].split(sep=',')
        sentiments = data["sentiment"][index].split(sep=',')
        for index2 in range(len(aspects)):
            if index2 >= len(sentiments):
                data["sentiment"][index] += "," + sentiments[0]
    return data


# --Filling NaN cells and Standardize notation
def adjusting(train_data, test_data):
    # --Fill NaN cells with none or NEU.
    train_data = train_data.fillna({'aspect': 'none'})
    test_data = test_data.fillna({'aspect': 'none'})
    train_data = train_data.fillna({'sentiment': 'NEU'})
    test_data = test_data.fillna({'sentiment': 'NEU'})
    train_data = train_data.fillna({'sentence': 'ok'})
    test_data = test_data.fillna({'sentence': 'ok'})

    # --Standardize notation to "POS, NEG, NEU"
    train_data = sentiment_unification(train_data)
    test_data = sentiment_unification(test_data)

    # --padding sentiment label
    train_data = padding_sentiment(train_data)
    test_data = padding_sentiment(test_data)

    # --return train data and test data
    return train_data, test_data


def text_lower(text):
    return text.lower()


def normalize_number(text):
    replaced_text = re.sub(r'¥d', '0', text)
    return replaced_text


def tokenize(text):
    return word_tokenize(text)


def stemming(text):
    stemmer = stem.PorterStemmer()
    for index in range(len(text)):
        text[index] = stemmer.stem(text[index])
    return text


def remove_stopwords(words):
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return words


# --Text preprocessing
# --text_lower and normalize_number are not used in the previous study
def data_preprocessing(data):
    data = copy.deepcopy(data)
    for index in range(len(data['sentence'])):
        # --not use
        # res_data['sentence'][index] = text_lower(res_data['sentence'][index])
        # res_data['sentence'][index] = normalize_number(res_data['sentence'][index])
        # --use
        data['sentence'][index] = tokenize(data['sentence'][index])
        data['sentence'][index] = stemming(data['sentence'][index])
        data['sentence'][index] = remove_stopwords(data['sentence'][index])
    data['sentence'] = [' '.join(text) for text in data['sentence']]
    return data


result_tokens = list()
result_rules = list()


# --Perform pattern matching and save the results
def get_pattern(data):
    # count = 0
    matcher = AspectMatcher()
    for index in range(len(data.sentence)):
        result = matcher.identify_aspect(copy.copy(data.sentence[index]))
        if result['aspect'] == '':
            result['aspect'] = 'none'
        data['aspect'][index] = result['aspect']
        if result['sentiment'] == '':
            result['sentiment'] = 'NEU'
        data['sentiment'][index] = result['sentiment']
        result_tokens.append(copy.deepcopy(matcher.get_tokens()))
        result_rules.append(copy.deepcopy(matcher.get_matched_rule()))
    # print("match:{}".format(count))
    return data


# --Split data containing multiple aspects into multiple data
def split_aspect(data, data_aspects, pattern_result):
    res = pd.DataFrame(columns=['sentence', 'aspect', 'sentiment', 'div_aspect', 'div_sentiment'])
    res2 = pd.DataFrame(columns=['aspect', 'sentiment'])
    res_size = 0
    for index in range(len(data)):
        div_aspects = data_aspects[index].split(sep=',')
        aspects = data["aspect"][index].split(sep=',')
        sentiments = data["sentiment"][index].split(sep=',')
        # print(div_aspects)
        # print(data["aspect"][index])
        # print(sentiments)
        for aspect in div_aspects:
            if aspect in aspects:
                i = aspects.index(aspect)
                d = [data["sentence"][index], data["aspect"][index], data["sentiment"][index], aspect, sentiments[i]]
                res.loc[res_size] = d
            else:
                d = [data["sentence"][index], data["aspect"][index], data["sentiment"][index], aspect, "NEU"]
                res.loc[res_size] = d
            res2.loc[res_size] = [pattern_result["aspect"][index], pattern_result["sentiment"][index]]
            res_size += 1
    return res, res2

