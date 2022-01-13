from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import copy
import numpy as np
import matplotlib.pyplot as plt


# --Get vector
# --return: train_data, test_data, sum of features
def vectorize(train_input, test_input):
    vectorizer = Tokenizer()
    vectorizer.fit_on_texts(train_input)
    train_input_vec = vectorizer.texts_to_sequences(train_input)
    test_input_vec = vectorizer.texts_to_sequences(test_input)
    train_input_vec = keras.preprocessing.sequence.pad_sequences(train_input_vec, padding='post')
    test_input_vec = keras.preprocessing.sequence.pad_sequences(test_input_vec, padding='post')
    return np.array(train_input_vec), np.array(test_input_vec), len(vectorizer.word_index) + 1


class LSTMModel:
    def __init__(self, train_input_vec, train_ans, test_input_vec, test_ans, fold=5):
        self.x_train = copy.deepcopy(train_input_vec)
        self.y_train = copy.deepcopy(train_ans)
        self.x_test = copy.deepcopy(test_input_vec)
        self.y_test = copy.deepcopy(test_ans)
        self.fold = fold
        self.model = None
        self.weights = list()
        self.result = None

    def create_model(self, feature_size, output_size):
        self.model = keras.Sequential()
        # --embedding layer
        self.model.add(Embedding(feature_size, 32, mask_zero=True))
        # --LSTM layer
        self.model.add(Bidirectional(LSTM(32, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(16)))
        # --fully connect layer
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dense(output_size, activation='softmax'))
        # --setting
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # --check summary
        # self.model.summary()

    def oversampling(self):
        from imblearn.over_sampling import SMOTE
        from imblearn.over_sampling import RandomOverSampler
        # --First, conduct oversampling
        ros = RandomOverSampler(random_state=23)
        self.x_train, self.y_train = ros.fit_resample(self.x_train, self.y_train)
        # --Second, conduct oversampling with SMOTE
        sm = SMOTE(random_state=67)
        # self.x_train, self.y_train = sm.fit_resample(self.x_train, self.y_train)

    def train(self, feature_size, output_size, filename):
        i = 0
        kf = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=0)
        for train_index, val_index in kf.split(self.x_train, self.y_train):
            print("Cross Validation: {}".format(i))
            self.create_model(feature_size, output_size)
            x_train = self.x_train[train_index]
            y_train = self.y_train[train_index]
            x_valid = self.x_train[val_index]
            y_valid = self.y_train[val_index]
            size = len(x_train)
            log = self.model.fit(x_train, y_train, epochs=100, batch_size=size//5,
                                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                          patience=10,
                                                                          verbose=1)],
                                 validation_data=(x_valid, y_valid))
            # --save weights of each layers
            w = list()
            for layer in self.model.layers:
                w.append(copy.deepcopy(layer.get_weights()))
            self.weights.append(w)
            # --print score
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
            plt.savefig(filename + "{}.png".format(i))
            #plt.show()
            plt.clf()
            i += 1
        return copy.deepcopy(self.weights)

    def test(self, filename):
        results = list()
        # --predict by each models
        for i in range(self.fold):
            # --load weights of each layers
            for l in range(len(self.model.layers)):
                self.model.layers[l].set_weights(self.weights[i][l])
            # --predict
            #predictions = self.model.predict_classes(self.x_test)
            predictions = self.model.predict(self.x_test)
            p_class = np.argmax(predictions, axis=1)
            results.append(predictions)
            print("model{}".format(i))
            cm = confusion_matrix(self.y_test, p_class)
            print(cm)
            print()
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
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print()
        np.savetxt(filename, cm, fmt='%d')
        return pred

