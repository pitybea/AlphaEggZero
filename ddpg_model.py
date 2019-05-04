#CopyRight no@none.not
from keras.layers import Input, Concatenate, BatchNormalization, add
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from keras import backend as K

def _neg_loss(y_true, y_pred):
    return -K.mean(y_pred, axis = -1)
    
class DDPGModel():
    def __init__(self, state_dim, action_dim, hidden1_dim = 400, hidden2_dim = 300):
        self.inp = Input(shape = (state_dim, ), name = 'inp')
        self.actor_hidden1 = Dense(hidden1_dim, activation = 'relu', name = 'actor_hid1')(self.inp)
        self.actor_norm1 = BatchNormalization(name = 'actor_norm1')(self.actor_hidden1)
        self.actor_hidden2 = Dense(hidden2_dim, activation = 'relu', name = 'actor_hid2')(self.actor_norm1)
        self.actor_norm2 = BatchNormalization(name = 'actor_norm2')(self.actor_hidden2)
        self.no_noise_actor = Dense(action_dim, activation = 'tanh', name = 'no_noise_actor')(self.actor_norm2)
        self.noise_inp = Input(shape = (actor_dim, ), name = 'noise')
        self.actor = add(name = 'actor')([self.no_noise_actor, self.noise_inp])
        
        self.critic_hidden1 = Dense(hidden1_dim, activation = 'relu', name = 'critic_hid1')(self.inp)
        self.critic_norm1 = BatchNormalization(name = 'critic_norm1')(self.critic_hidden1)
        self.actor_middle = Dense(hidden2_dim, activation = 'relu', name = 'actor_middle')(self.actor)
        self.critic_hidden2 = Dense(hidden2_dim, activation = 'relu', name = 'critic_hid2')(Concatenate()([self.actor_middle, self.critic_norm1]))
        self.critic_norm2 = BatchNormalization(name = 'critic_norm2')(self.critic_hidden2)
        self.critic = Dense(1, activation = 'tanh', name = 'critic')(self.critic_norm2)

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
        self.model.fit(data_label[0], data_label[1], batch_size = 5, epochs = 10, verbose = 0)
            
    def train_actor_net(self, data_label):
        self.model = Model(inputs = self.inp, outputs = self.critic)
        self.model.get_layer('hid1').trainable = True
        self.model.get_layer('actor').trainable = True
        self.model.get_layer('hid2').trainable = False
        self.model.get_layer('hid3').trainable = False
        self.model.get_layer('critic').trainable = False
        self.model.compile(loss = _neg_loss, optimizer = self.sgd)
        self.model.fit(data_label[0], data_label[1], batch_size = 5, epochs = 3, verbose = 0)
        

    def get_model_detail(self):
        data = np.eye(self.egg_total)
        actions = Model(inputs = self.inp, outputs = self.actor).predict(data)
        hd2_oup = Model(inputs = self.inp, outputs = self.hidden2).predict(data)
        hd3_oup = Model(inputs = self.inp, outputs = self.hidden3).predict(data)
        return actions, hd2_oup, hd3_oup
        
