#CopyRight no@none.not
from keras.layers import Input, Concatenate
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from keras import backend as K


class TwoHeadModel():
    def __init__(self, egg_total, max_egg_per_round):
        inp = Input(shape = (egg_total, ))
        hidden = Dense(egg_total, activation = 'relu')(inp)
        drpout = Dropout(0.6)(hidden)
        oup1 = Dense(1, activation = 'tanh')(drpout)
        oup2 = Dense(max_egg_per_round, activation = 'softmax')(drpout)
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

    def train_model(self, data_label):
        self.model.fit(data_label[0], data_label[1], batch_size = 5, epochs = 12)

    def get_status(self):
        actions = [np.argmax(self.get_action_posibility(i)) + 1
                   for i in range(1, self.egg_total + 1)]
        win_loses = [self.get_win_lose(i) for i in range(1, self.egg_total + 1)]
        return actions, win_loses
        

def _neg_loss(y_true, y_pred):
    return -K.mean(y_pred, axis = -1)
    
class DDPGModel():
    def __init__(self, egg_total, max_egg_per_round):
        self.inp = Input(shape = (egg_total, ))
        self.hidden1 = Dense(egg_total, activation = 'relu')(inp)
        drpout = Dropout(0.6)(self.hidden1)
        self.actor = Dense(1, activation = 'tanh')(drpout)
        self.hidden2 = Dense(egg_total / 2 + 2, activation = 'relu')(self.inp)
        self.critic = Dense(1, activation = 'tanh')(Concatenate()([self.actor, self.hidden2]))
        self.sgd = SGD(lr = 0.1)

        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round

    def to_actor_net(self):
        self.model = Model(inputs = self.inp, outputs = self.actor)

    def to_critic_net(self):
        self.model = Model(inputs = self.inp, outputs = self.actor)
        
    def to_critic_training_net(self):
        self.hidden1.trainable = False
        self.actor.trainable = False
        self.hidden2.trainable = True
        self.critic.trainable = True
        self.model = Model(inputs = self.inp, outputs = self.critic)
        self.model.compile(loss = 'mse', optimizer = self.sgd)

    def to_actor_training_net(self):
        self.hidden1.trainable = True
        self.actor.trainable = True
        self.hidden2.trainable = False
        self.critic.trainable = False
        self.model = Model(inputs = self.inp, outputs = self.critic)
        self.model.compile(loss = _neg_loss, optimizer = self.sgd)

    def get_action(self, egg_leftover):
        assert egg_leftover <= self.egg_total
        assert egg_leftover >= 1
        arr = np.zeros((1, self.egg_total))
        arr[0][egg_leftover - 1] = 1.0
        self.to_actor_net()
        pred = self.model.predict(arr)[0]
        a = 0.5 * (self.max_egg_per_round - 1)
        b = 0.5 * (self.max_egg_per_round + 1)
        return int(a * pred + b)

    
