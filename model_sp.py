from tensorflow import keras
import tensorflow as tf
from keras.layers import LSTM, Embedding, Dense, Bidirectional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import gensim


PATTERN_USE = True
ASPECT_USE = False
OVER_SAMPLING = True
UNDER_SAMPLING = False

# LSTM_LAYER = [512, 256]
# DENSE_LAYER = [256, 64, 16]
# LSTM_LAYER = [64, 32]
# DENSE_LAYER = [32, 16]
LSTM_LAYER = [32, 16]
DENSE_LAYER = [16, 16]
EARLY_STOP = 10
BATCH_SIZE = 32


def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()


def extract_df(df, index):
    columns = df.columns
    a = df.values
    res = a[index]
    return pd.DataFrame(res, columns=columns)


def model_compile(model):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# --add LSTM layer
def add_lstm(model, feature_size, vectorizer, wv):
    # --embedding layer
    if vectorizer is None:
        model.add(Embedding(feature_size, LSTM_LAYER[0], mask_zero=True))
    else:
        null_word = np.zeros(300)
        char_indices = vectorizer.word_index
        indices_char = dict([(value, key) for (key, value) in char_indices.items()])
        embedding_matrix = np.zeros((len(char_indices)+1, 300))
        for i, word in indices_char.items():
            try:
                embedding_matrix[i] = wv[word]
            except:
                embedding_matrix[i] = null_word
        embedding_matrix[0] = null_word
        model.add(Embedding(len(char_indices)+1, 300, weights=[embedding_matrix], mask_zero=True, trainable=False))
    # --LSTM layer
    for size in LSTM_LAYER:
        model.add(Bidirectional(LSTM(size, return_sequences=True)))
    # model.add(Bidirectional(LSTM(32, return_sequences=True)))
    # model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(LSTM_LAYER[-1])))
    return model


# --add First Dense layer
def add_first_dense(model, input_shape):
    model.add(Dense(units=DENSE_LAYER[0], activation='relu', input_shape=input_shape))
    return model


# --add Dense layer
def add_dense(model, output_size, activation):
    model.add(Dense(output_size, activation=activation))
    return model


# --Add the results of pattern matching as features.
def add_pattern_feature(input_vec, pattern_result, aspects, sentiments):
    if pattern_result is None:
        return input_vec
    if False:
        # --prepare pair of aspect and sentiment
        pair = list()
        for aspect in aspects:
            for sentiment in sentiments:
                pair.append(aspect + sentiment)

        # --Convert to an array to add features
        print("add pattern feature")
        # print(type(input_vec))
        # input_vec = input_vec.toarray()

        # --For each label pair, check the results of pattern matching.
        for index1 in range(len(pair)):
            # --First, add a zero to the end.
            input_vec = np.insert(input_vec, input_vec.shape[1], 0, axis=1)
            for index2 in range(len(input_vec)):
                pred_aspects = pattern_result.aspect[index2].split(sep=' ')
                pred_sentiments = pattern_result.sentiment[index2].split(sep=' ')
                pred_pair = list()
                for index3 in range(len(pred_aspects)):
                    a = pred_aspects[index3]
                    s = 'NEU'
                    if index3 < len(pred_sentiments):
                        s = pred_sentiments[index3]
                    pred_pair.append(a + s)
                if pair[index1] in pred_pair:
                    # --Change the added 0 to 1
                    input_vec[index2][input_vec.shape[1] - 1] = 1
    else:
        # not sentiment but only aspect.
        # --Convert to an array to add features
        print("add pattern feature")

        # --For each label pair, check the results of pattern matching.
        for index1 in range(len(aspects)):
            # --First, add a zero to the end.
            input_vec = np.insert(input_vec, input_vec.shape[1], 0, axis=1)
            for index2 in range(len(input_vec)):
                pred_aspects = pattern_result.aspect[index2].split(sep=' ')
                if aspects[index1] in pred_aspects:
                    # --Change the added 0 to 1
                    input_vec[index2][input_vec.shape[1] - 1] = 1

    # --Restore input_vec to its original type
    # input_vec = csr_matrix(input_vec)
    # --return input_vec
    return input_vec


# --Add the results of aspect classification as features.
def add_aspect_feature(input_vec, aspect_feature, aspects):
    if aspect_feature is None:
        return input_vec
    # --Convert to an array to add features
    print("add aspect feature")
    # print(type(input_vec))
    # input_vec = input_vec.toarray()
    # --For each aspect
    for index in range(len(aspects)):
        # --First, add a zero to the end.
        input_vec = np.insert(input_vec, input_vec.shape[1], 0, axis=1)
        # --For each data, add a feature.
        for index2 in range(len(input_vec)):
            # --If the aspect is predicted
            if aspects[index] in aspect_feature[index2]:
                # --Change the added 0 to 1
                input_vec[index2][input_vec.shape[1] - 1] = 1
    # --Restore input_vec to its original type
    # input_vec = csr_matrix(input_vec)
    # --return input_vec
    return input_vec


# --plot and print result of Training
def print_score(log, filename):
    loss = log.history['loss']
    val_loss = log.history['val_loss']
    acc = log.history['accuracy']
    val_acc = log.history['val_accuracy']
    print('result:')
    print(loss)
    print(val_loss)
    print(acc)
    print(val_acc)
    # グラフ表示
    plt.plot(log.history['loss'], label='loss')
    plt.plot(log.history['val_loss'], label='val_loss')
    plt.legend(frameon=False)
    plt.xlabel("epochs")
    plt.ylabel("crossentropy")
    plt.savefig(filename)
    # plt.show()
    plt.clf()


def oversampling(x_train, y_train):
    from imblearn.over_sampling import RandomOverSampler
    # --conduct oversampling
    ros = RandomOverSampler(random_state=23)
    x_train, y_train = ros.fit_resample(x_train, y_train)
    return x_train, y_train


def undersampling(x_train, y_train):
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=2)
    x_train, y_train = nm.fit_sample(x_train, y_train)
    return x_train, y_train


class LSTMModelSP:
    def __init__(self, train_input_vec, train_ans, test_input_vec, test_ans,
                 train_pattern_result, test_pattern_result, train_aspect, test_aspect, aspects, fold, vectorizer=None):
        self.x_train = copy.deepcopy(train_input_vec)
        self.y_train = copy.deepcopy(train_ans)
        self.x_test = test_input_vec
        self.y_test = test_ans
        self.fold = fold
        # --model have LSTM and Dense layer for training weights of LSTM layer
        self.model = None
        # --LSTM model and Dense model
        self.lstm_model = None
        self.dense_model = None
        # --weights[i][j] -> weight of [fold i, layer j]
        self.weights = list()
        self.dense_weights = list()
        self.result = None
        # --result of pattern matching
        self.train_pattern_result = train_pattern_result
        self.test_pattern_result = test_pattern_result
        # --result of aspect classification
        self.train_aspect = train_aspect
        self.test_aspect = test_aspect
        # --labels of aspects and sentiments
        self.aspects = aspects
        self.sentiments = ["POS", "NEG", "NEU"]
        self.dense_input_size = 0
        self.vectorizer = vectorizer
        self.wv = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def train_lstm(self, feature_size, output_size, x_train, y_train, x_valid, y_valid):
        # --create model
        self.model = keras.Sequential()
        # --embedding and latm layer
        self.model = add_lstm(self.model, feature_size, self.vectorizer, self.wv)
        # --fully connect layer
        self.model = add_dense(self.model, DENSE_LAYER[0], 'relu')
        for size in DENSE_LAYER:
            self.model = add_dense(self.model, size, 'relu')
        self.model = add_dense(self.model, output_size, 'softmax')
        # --setting
        self.model = model_compile(self.model)
        # --check summary
        # self.model.summary()
        # self.create_model(feature_size, output_size)
        size = len(x_train)
        batch_size = BATCH_SIZE
        print("batch_size = {}".format(batch_size))
        x_train2 = copy.deepcopy(x_train)
        y_train2 = copy.deepcopy(y_train)
        if UNDER_SAMPLING is True:
            x_train2, y_train2 = undersampling(x_train2, y_train2)
            print("under_sampling: {} -> {}".format(len(x_train), len(x_train2)))
        log = self.model.fit(x_train, y_train, epochs=150, batch_size=batch_size,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=EARLY_STOP,
                                                                      verbose=1)],
                             validation_data=(x_valid, y_valid))
        keras.backend.clear_session()
        gc.collect()
        # --save weights of each layers
        w = list()
        for layer in self.model.layers:
            w.append(copy.deepcopy(layer.get_weights()))
        self.weights.append(w)
        del self.model
        gc.collect()
        return log

    def create_lstm(self, fold_num, feature_size):
        # --create model
        self.lstm_model = keras.Sequential()
        # --create each layer
        self.lstm_model = add_lstm(self.lstm_model, feature_size, self.vectorizer, self.wv)
        # --compile
        self.lstm_model = model_compile(self.lstm_model)
        # --load weights
        for i in range(len(self.lstm_model.layers)):
            self.lstm_model.layers[i].set_weights(self.weights[fold_num][i])

    def train_dense(self, fold_num, feature_size, output_size, x_train, y_train, pattern_train, aspect_train,
                    x_valid, y_valid, pattern_valid, aspect_valid):
        # --create lstm model for creating input vector
        # --create model
        self.create_lstm(fold_num, feature_size)
        # --create input vector
        input_train = self.lstm_model.predict(x_train)
        input_valid = self.lstm_model.predict(x_valid)
        if ASPECT_USE is True:
            input_train = add_aspect_feature(input_train, aspect_train, self.aspects)
            input_valid = add_aspect_feature(input_valid, aspect_valid, self.aspects)
        if PATTERN_USE is True:
            input_train = add_pattern_feature(input_train, pattern_train, self.aspects, self.sentiments)
            input_valid = add_pattern_feature(input_valid, pattern_valid, self.aspects, self.sentiments)
        # if UNDER_SAMPLING is True:
        #     input_train, y_train = undersampling(input_train, y_train)
        if OVER_SAMPLING is True:
            input_train, y_train = oversampling(input_train, y_train)
        # --create Dense only model
        self.dense_model = keras.Sequential()
        self.dense_input_size = input_train.shape[1]
        self.dense_model = add_first_dense(self.dense_model, (self.dense_input_size,))
        for size in DENSE_LAYER:
            self.dense_model = add_dense(self.dense_model, size, 'relu')
        self.dense_model = add_dense(self.dense_model, output_size, 'softmax')
        self.dense_model = model_compile(self.dense_model)
        # self.dense_model.summary()
        # --train
        # size = len(x_train)
        batch_size = BATCH_SIZE
        print("batch_size = {}".format(batch_size))
        log = self.dense_model.fit(input_train, y_train, epochs=100, batch_size=batch_size,
                                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                            patience=EARLY_STOP,
                                                                            verbose=1)],
                                   validation_data=(input_valid, y_valid))
        keras.backend.clear_session()
        gc.collect()
        # --save weights of dense layers
        w = list()
        for layer in self.dense_model.layers:
            w.append(copy.deepcopy(layer.get_weights()))
        self.dense_weights.append(w)
        del self.lstm_model
        del self.dense_model
        gc.collect()
        return log

    def create_dense(self, fold_num, input_size, output_size):
        # --create model
        self.dense_model = keras.Sequential()
        # --create dense layer
        self.dense_model = add_first_dense(self.dense_model, (input_size,))
        for size in DENSE_LAYER:
            self.dense_model = add_dense(self.dense_model, size, 'relu')
        self.dense_model = add_dense(self.dense_model, output_size, 'softmax')
        # self.dense_model.summary()
        # --load weights
        for i in range(len(self.dense_model.layers)):
            self.dense_model.layers[i].set_weights(self.dense_weights[fold_num][i])

    # --Train!!
    def train(self, feature_size, output_size, filename):
        tf.random.set_seed(23)
        np.random.seed(67)
        i = 0
        kf = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=0)
        for train_index, val_index in kf.split(self.x_train, self.y_train):
            # --train data
            x_train = self.x_train[train_index]
            y_train = self.y_train[train_index]
            pattern_train = extract_df(self.train_pattern_result, train_index)
            aspect_train = None
            if not (self.train_aspect is None):
                aspect_train = self.train_aspect.values[train_index]
            # --valid data
            x_valid = self.x_train[val_index]
            y_valid = self.y_train[val_index]
            aspect_valid = None
            pattern_valid = extract_df(self.train_pattern_result, val_index)
            if not (self.train_aspect is None):
                aspect_valid = self.train_aspect.values[val_index]

            # --Train LSTM layer
            print("train LSTM layer")
            log = self.train_lstm(feature_size, output_size, x_train, y_train, x_valid, y_valid)
            # --print score
            print_score(log, filename + "LSTM{}.png".format(i))

            # --Train Dense layer
            print("train Dense layer")
            log = self.train_dense(i, feature_size, output_size,
                                   x_train, y_train, pattern_train, aspect_train,
                                   x_valid, y_valid, pattern_valid, aspect_valid)
            print_score(log, filename + "Dense{}.png".format(i))
            i += 1
            gc.collect()

    # --Test!!
    def test(self, feature_size, output_size, filename):
        results = list()
        # --predict by each models
        for i in range(self.fold):
            # --load weights of each layers
            self.create_lstm(i, feature_size)
            self.create_dense(i, self.dense_input_size, output_size)
            # --predict
            input_vec = self.lstm_model.predict(self.x_test)
            if ASPECT_USE is True:
                input_vec = add_aspect_feature(input_vec, self.test_aspect, self.aspects)
            if PATTERN_USE is True:
                input_vec = add_pattern_feature(input_vec, self.test_pattern_result, self.aspects, self.sentiments)
            predictions = self.dense_model.predict(input_vec)
            p_class = np.argmax(predictions, axis=1)
            results.append(predictions)
            print("model{}".format(i))
            cm = confusion_matrix(self.y_test, p_class)
            print(cm)
            print()
            del self.lstm_model
            del self.dense_model
            gc.collect()
        # --calculate average of prediction of each model
        self.result = copy.deepcopy(results[0])
        pred = list()
        for i in range(1, self.fold):
            for j in range(len(self.result)):
                self.result[j] += results[i][j]
        for i in range(len(self.result)):
            self.result[i] /= self.fold
            pred.append(np.argmax(self.result[i]))
            # print("{} : {}".format(self.result[i], self.y_test[i]))
        cm = confusion_matrix(self.y_test, pred)
        print(cm)
        precision = 0
        recall = 0
        if len(cm) == 3:
            if cm[0][0] > 0:
                precision = cm[0][0] / (cm[0][0] + cm[1][0])
                recall = cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])
            print("POS:precision: {}".format(precision))
            print("POS:recall: {}".format(recall))
            if cm[1][1] > 0:
                precision = cm[1][1] / (cm[1][1] + cm[0][1])
                recall = cm[1][1] / (cm[1][1] + cm[1][0] + cm[1][2])
            print("NEG:precision: {}".format(precision))
            print("NEG:recall: {}".format(recall))
            print()
        else:
            if cm[1][1] > 0:
                precision = cm[1][1] / (cm[1][1] + cm[0][1])
                recall = cm[1][1] / (cm[1][1] + cm[1][0])
            print("precision: {}".format(precision))
            print("recall: {}".format(recall))
            print()
        np.savetxt(filename, cm, fmt='%d')
        return pred, precision, recall

    # train_steps, train_batches = batch_iter(x_train2, y_train2, batch_size)
    # valid_steps, valid_batches = batch_iter(x_valid, y_valid, batch_size)
    # self.model.fit_generator(train_batches, train_steps,
    #                          epochs=100,
    #                          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                                   patience=10,
    #                                                                   verbose=1)],
    #                          validation_data=valid_batches,
    #                          validation_steps=valid_steps)

    # train_steps, train_batches = batch_iter(input_train, y_train, batch_size)
    # valid_steps, valid_batches = batch_iter(input_valid, y_valid, batch_size)
    # self.dense_model.fit_generator(train_batches, train_steps,
    #                                epochs=100,
    #                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                                         patience=10,
    #                                                                         verbose=1)],
    #                                validation_data=valid_batches,
    #                                validation_steps=valid_steps)
