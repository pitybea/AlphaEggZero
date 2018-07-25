#CopyRight no@none.not
from keras.layers import Input
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score, average_precision_score


class TwoHeadModel():
    def __init__(self, egg_total, max_egg_per_round):
        inp = Input(shape = (egg_total, ))
        hidden = Dense(int(egg_total * 1.5), activation = 'relu')(inp)
        oup1 = Dense(1, activation = 'tanh')(hidden)
        oup2 = Dense(max_egg_per_round, activation = 'softmax')(hidden)
        self.model = Model(inputs = inp, outputs = [oup1, oup2])

        sgd = SGD(lr = 0.1)
        self.model.compile(loss = ['mse', 'categorical_crossentropy'], optimizer = sgd)
        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round


    def __get_predict(self, egg_leftover):
        assert egg_leftover <= self.egg_total
        assert egg_leftover >= 1
        arr = np.zeros((1, self.egg_total))
        arr[0][egg_leftover - 1] = 1.0
        return self.model.predict(arr)
        
    def get_win_lose(self, egg_leftover):
        return self.__get_predict(egg_leftover)[0][0][0]

    def get_action_posibility(self, egg_leftover):
        return self.__get_predict(egg_leftover)[1][0]

    def train_model(self, data, lebel):
        self.model.fit(data, label, batch_size = 20)
        #[array([[1], [1]]), array([[0.2, 0.3], [0.3, 0.1]])]
        
    
        
