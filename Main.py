from Load import load_dataset
from model import vectorize, LSTMModel
from model_sp import LSTMModelSP
import preprocessing
from preprocessing import split_aspect

import copy
import gc


# --Change all labels except <aspect> to 'none'.
def remove_aspect(aspect, train_ans, test_ans):
    for index in range(len(train_ans)):
        if aspect in train_ans[index]:
            train_ans[index] = 1
        else:
            train_ans[index] = 0
    for index in range(len(test_ans)):
        if aspect in test_ans[index]:
            test_ans[index] = 1
        else:
            test_ans[index] = 0


# --[POS, NEG, NEU] -> [0, 1, 2]
def remove_sentiment(labels, train_ans, test_ans):
    for index in range(len(train_ans)):
        train_ans[index] = labels.index(train_ans[index])
    for index in range(len(test_ans)):
        test_ans[index] = labels.index(test_ans[index])


# --for example, label = [none, community], prediction = [1, 0, 0] -> [community, none, none]
def get_prediction(prediction, label):
    res = list()
    for i in range(len(prediction)):
        res.append(label[prediction[i]])
    return res


def push_result(result_data, prediction):
    for index in range(len(result_data)):
        if 'none' in prediction[index] or 'NEU' in prediction[index]:
            continue
        if 'none' in result_data[index] or 'NEU' in result_data[index]:
            result_data[index] = prediction[index]
        else:
            result_data[index] += ',{}'.format(prediction[index])


# remove data the result of aspect prediction is 'none'
def remove_none(data, pattern_result, result_data):
    data['pred_aspect'] = 'none'
    for index in range(len(data)):
        if result_data['aspect_prediction'][index] == 'none':
            data = data.drop(index)
            pattern_result = pattern_result.drop(index)
            result_data = result_data.drop(index)
        else:
            data['pred_aspect'][index] = result_data['aspect_prediction'][index]
    data = data.reset_index(drop=True)
    result_data = result_data.reset_index(drop=True)
    pattern_result = pattern_result.reset_index(drop=True)
    return data, pattern_result, result_data


def separate_by_aspect(train_data, test_data, aspect_prediction, train_pattern_result, test_pattern_result):
    # --Separate by aspect
    train_data, train_pattern_result = split_aspect(train_data, train_data.aspect, train_pattern_result)
    test_data, test_pattern_result = split_aspect(test_data, aspect_prediction, test_pattern_result)
    return train_data, test_data, train_pattern_result, test_pattern_result


# ###################### #
# --For Cross validation
FOLD = 5
# --if I use amazon data, AMAZON is True
AMAZON = True


# model = LSTMModelSP(x_train, y_train, x_test, y_test, train_pattern_result, test_pattern_result,
#                     None, None,
#                     aspects, fold=FOLD)
# prediction = model.test(word_sum, len(labels), save_path + aspect + ".csv")


def main():
    save_dir = 'normal'
    save_path = ''
    train_path = ''
    test_path = ''
    aspects = list()
    # --aspects of products
    if AMAZON is True:
        aspects = ['cost', 'community', 'compatibility', 'functional', 'looks',
                   'performance', 'reliability', 'usability']
        train_path = "data/amazon_us_pc_training.csv"
        test_path = "data/amazon_us_pc_test.csv"
        save_path = 'result/amazon/' + save_dir + '/'
    else:
        aspects = ['community', 'compatibility', 'documentation', 'functional',
                   'performance', 'reliability', 'usability']
        train_path = "data/training.csv"
        test_path = "data/data_rq1_rq2.csv"
        save_path = 'result/api/' + save_dir + '/'

    # --load datasets
    # --train data
    train_data = load_dataset(train_path)
    # --test data
    test_data = load_dataset(test_path)

    # ##### preprocessing ##### #
    # --Filling NaN cells and Standardize notation
    train_data, test_data = preprocessing.adjusting(train_data, test_data)

    # --prepare result data
    result_data = copy.copy(test_data)
    result_data['aspect_prediction'] = 'none'
    result_data['sentiment_prediction'] = 'NEU'

    # --Get the result of Pattern-Matching.
    print('Conduct Pattern-Matching Now...')
    train_pattern_result = preprocessing.get_pattern(copy.copy(train_data))
    test_pattern_result = preprocessing.get_pattern(copy.copy(test_data))

    # --text_lower and normalize_number are not used in the previous study
    preprocessed_train_data = preprocessing.data_preprocessing(copy.deepcopy(train_data))
    preprocessed_test_data = preprocessing.data_preprocessing(copy.deepcopy(test_data))

    # vectorize
    train_input_vec, test_input_vec, word_sum = vectorize(preprocessed_train_data["sentence"].tolist(),
                                                          preprocessed_test_data["sentence"].tolist())

    # ###### Aspect classification ######## #
    for aspect in aspects:
        print(aspect)
        labels = ["none", aspect]
        # --create data
        x_train = copy.deepcopy(train_input_vec)
        x_test = copy.deepcopy(test_input_vec)
        y_train = copy.deepcopy(preprocessed_train_data["aspect"]).values
        y_test = copy.deepcopy(preprocessed_test_data["aspect"]).values
        # --Change all labels except <aspect> to 'none'.
        remove_aspect(aspect, y_train, y_test)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # --Create model and train
        #model = LSTMModel(x_train, y_train, x_test, y_test, fold=FOLD)
        model = LSTMModelSP(x_train, y_train, x_test, y_test, train_pattern_result, test_pattern_result,
                            None, None,
                            aspects, fold=FOLD)
        # over sampling
        #model.oversampling()
        # --Train
        model.train(word_sum, len(labels), save_path + aspect)
        # --Predicting test data
        # prediction = model.test(save_path + aspect + ".csv")
        prediction = model.test(word_sum, len(labels), save_path + aspect + ".csv")
        prediction = get_prediction(prediction, labels)
        # --add result data
        push_result(result_data['aspect_prediction'], prediction)
        del model

        gc.collect()

    # #### data split #### #
    # --remove the data which is predicted as "none"
    preprocessed_test_data, test_pattern_result, result_data = remove_none(preprocessed_test_data,
                                                                           test_pattern_result, result_data)
    print("Number of data: {}".format(len(preprocessed_test_data)))
    # --separate data by aspect
    preprocessed_train_data, preprocessed_test_data, train_pattern_result, test_pattern_result = separate_by_aspect(
        preprocessed_train_data, preprocessed_test_data, result_data["aspect_prediction"],
        train_pattern_result, test_pattern_result)

    # ########### Sentiment classification ############### #
    labels = ['POS', 'NEG', 'NEU']
    # --vectorize
    train_input_vec, test_input_vec, word_sum = vectorize(preprocessed_train_data["sentence"].tolist(),
                                                          preprocessed_test_data["sentence"].tolist())
    # --create data
    x_train = copy.deepcopy(train_input_vec)
    x_test = copy.deepcopy(test_input_vec)
    y_train = copy.deepcopy(preprocessed_train_data["div_sentiment"]).values
    y_test = copy.deepcopy(preprocessed_test_data["div_sentiment"]).values
    # --Change all labels int.
    remove_sentiment(labels, y_train, y_test)
    # --convert
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    # --Create model and train
    model = LSTMModelSP(x_train, y_train, x_test, y_test, train_pattern_result, test_pattern_result,
                        preprocessed_train_data['div_aspect'], preprocessed_test_data['div_sentiment'],
                        aspects, fold=FOLD)
    # --over sampling
    #model.oversampling()
    # --Train
    model.train(word_sum, len(labels), save_path + "sentiment")
    # --Predicting test data
    model.test(word_sum, len(labels), save_path + "sentiment.csv")


if __name__ == '__main__':
    main()

