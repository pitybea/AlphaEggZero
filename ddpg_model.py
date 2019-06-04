#CopyRight no@none.not
from keras.layers import Input, Concatenate, BatchNormalization, Add
from keras.layers.core import Dense
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.layers.recurrent import SimpleRNN



def _neg_loss(y_true, y_pred):
    return -K.mean(y_pred, axis = -1)
    
class DDPGModel():
    def __init__(self, state_dim, action_dim, hidden1_dim = 400, hidden2_dim = 300):
        initializer = RandomUniform(minval = -0.003, maxval = 0.003)
        self.inp = Input(shape = (state_dim, ), name = 'inp')
        actor_hidden1 = Dense(hidden1_dim, activation = 'relu', name = 'actor_hid1', kernel_initializer = initializer)(self.inp)
        actor_norm1 = BatchNormalization(name = 'actor_norm1')(actor_hidden1)
        actor_hidden2 = Dense(hidden2_dim, activation = 'relu', name = 'actor_hid2', kernel_initializer = initializer)(actor_norm1)
        actor_norm2 = BatchNormalization(name = 'actor_norm2')(actor_hidden2)
        no_noise_actor = Dense(action_dim, activation = 'tanh', name = 'no_noise_actor', kernel_initializer = initializer)(actor_norm2)
        self.noise_inp = Input(shape = (action_dim, ), name = 'noise')
        self.actor = Add(name = 'actor')([no_noise_actor, self.noise_inp])
        
        critic_hidden1 = Dense(hidden1_dim, activation = 'relu', name = 'critic_hid1', kernel_initializer = initializer)(self.inp)
        critic_norm1 = BatchNormalization(name = 'critic_norm1')(critic_hidden1)
        actor_middle = Dense(hidden2_dim, activation = 'relu', name = 'actor_middle', kernel_initializer = initializer)(self.actor)
        critic_hidden2 = Dense(hidden2_dim, activation = 'relu', name = 'critic_hid2', kernel_initializer = initializer)(Concatenate()([actor_middle, critic_norm1]))
        critic_norm2 = BatchNormalization(name = 'critic_norm2')(critic_hidden2)
        self.critic = Dense(1, activation = 'tanh', name = 'critic', kernel_initializer = initializer)(critic_norm2)

        self.action_model = Model(inputs = [self.inp, self.noise_inp], outputs = self.actor)
        self.model = Model(inputs = [self.inp, self.noise_inp], outputs = self.critic)
        self.critic_settings = {'actor_hid1': False, 'actor_norm1': False, 'actor_hid2': False, 'actor_norm2': False,'no_noise_actor': False, 'actor': False,
                                'critic_hid1': True, 'critic_norm1': True, 'actor_middle': True, 'critic_hid2': True, 'critic_norm2': True, 'critic': True}
        critic_lr = 4e-3
        actor_lr = 7e-3
        self.critic_optimizer = Adam(lr = critic_lr)
        self.actor_optimizer = Adam(lr = actor_lr)

    def to_critic_net(self):
        for s in self.critic_settings:
            self.model.get_layer(s).trainable = self.critic_settings[s]
        self.model.compile(loss = 'mse', optimizer = self.critic_optimizer)
                
    def to_actor_net(self):
        for s in self.critic_settings:
            self.model.get_layer(s).trainable = (not self.critic_settings[s])
        self.model.compile(loss = _neg_loss, optimizer = self.actor_optimizer)
        
    def describe(self):
        self.model.summary()
        for i in self.model.layers:
            print(i.name, i.trainable)
    
    def get_action(self, inp_data, noise_data):
        return self.action_model.predict([inp_data, noise_data])

    def get_critic(self, inp_data, noise_data):
        return self.model.predict([inp_data, noise_data])
    
    def train_model(self, inp_data, label):
        noise_data = np.zeros([inp_data.shape[0], 1])
        self.to_critic_net()
        self.model.fit([inp_data, noise_data], label, batch_size = 64, epochs = 10)
        self.to_actor_net()
        self.model.fit([inp_data, noise_data], label, batch_size = 64, epochs = 20)
