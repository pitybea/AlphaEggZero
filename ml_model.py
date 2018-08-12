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
        self.inp = Input(shape = (egg_total, ), name = 'inp')
        self.hidden1 = Dense(egg_total, activation = 'relu', name = 'hid1')(self.inp)
        drpout1 = Dropout(0.6)(self.hidden1)
        self.actor = Dense(1, activation = 'tanh', name = 'actor')(drpout1)
        self.hidden2 = Dense(egg_total / 2 + 2, activation = 'relu', name = 'hid2')(self.inp)
        self.hidden3 = Dense(egg_total / 2 + 3, activation = 'relu', name = 'hid3')(Concatenate()([self.actor, self.hidden2]))
        drpout2 = Dropout(0.6)(self.hidden3)
        self.critic = Dense(1, activation = 'tanh', name = 'critic')(drpout2)
        self.sgd = SGD(lr = 0.1)

        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round
        self.a = 0.5 * (self.max_egg_per_round - 1)
        self.b = 0.5 * (self.max_egg_per_round + 1)

        
    def to_critic_net(self):
        self.model = Model(inputs = self.inp, outputs = self.critic)
        
    def to_actor_net(self):
        self.model = Model(inputs = self.inp, outputs = self.actor)

    def train_critic_net(self, data_label):
        self.model = Model(inputs = self.inp, outputs = self.critic)
        self.model.get_layer('hid1').trainable = False
        self.model.get_layer('actor').trainable = False
        self.model.get_layer('hid2').trainable = True
        self.model.get_layer('hid3').trainable = True
        self.model.get_layer('critic').trainable = True
        self.model.compile(loss = 'mse', optimizer = self.sgd)
        self.model.fit(data_label[0], data_label[1], batch_size = 5, epochs = 30, verbose = 0)
            
    def train_actor_net(self, data_label):
        self.model = Model(inputs = self.inp, outputs = self.critic)
        self.model.get_layer('hid1').trainable = True
        self.model.get_layer('actor').trainable = True
        self.model.get_layer('hid2').trainable = False
        self.model.get_layer('hid3').trainable = False
        self.model.get_layer('critic').trainable = False
        self.model.compile(loss = _neg_loss, optimizer = self.sgd)
        self.model.fit(data_label[0], data_label[1], batch_size = 5, epochs = 3, verbose = 0)
        
    def get_action(self, egg_leftover, explore = False):
        assert egg_leftover <= self.egg_total
        assert egg_leftover >= 1
        arr = np.zeros((1, self.egg_total))
        arr[0][egg_leftover - 1] = 1.0
        self.to_actor_net()
        pred = self.model.predict(arr)[0][0]
        if explore:
            pred += -1 ** np.random.choice([1, 2]) * np.random.rand() * 0.6
        return int(self.a * pred + self.b)

    def get_critic(self, egg_leftover):
        assert egg_leftover <= self.egg_total
        assert egg_leftover >= 1
        arr = np.zeros((1, self.egg_total))
        arr[0][egg_leftover - 1] = 1.0
        self.to_critic_net()
        pred = self.model.predict(arr)[0]
        return pred
    
    def get_status(self):
        actions = [self.get_action(i) for i in range(1, self.egg_total + 1)]
        win_loses = [self.get_critic(i)[0] for i in range(1, self.egg_total + 1)]
        return actions, win_loses

    def get_model_detail(self):
        data = np.eye(self.egg_total)
        actions = Model(inputs = self.inp, outputs = self.actor).predict(data)
        hd2_oup = Model(inputs = self.inp, outputs = self.hidden2).predict(data)
        hd3_oup = Model(inputs = self.inp, outputs = self.hidden3).predict(data)
        return actions, hd2_oup, hd3_oup
        
