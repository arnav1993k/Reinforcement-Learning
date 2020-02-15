import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
class Agent_NN:
    def __init__(self, nS, nA, epsilon = 1, decay = 0.99, batch_size = 50, gamma = 0.99):
        self.nS = nS
        self.nA = nA
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = decay
        self.gamma = gamma
        self.epochs = 1
        self.verbose = 0
        self.batch_size = batch_size
        self.memory = deque(maxlen=5000)
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.loss = deque(maxlen=100)

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        return model

    def add_memory(self, curr_state, action, reward, next_state, done):
        self.memory.append((curr_state, action, reward, next_state, done))

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def action(self, curr_state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        q = self.model.predict(curr_state)
        return np.argmax(q[0])

    def train(self):

        minibatch = random.sample(self.memory, self.batch_size)
        minibatch = np.array(minibatch)
        non_terminating = np.where(minibatch[:, 4] == False)
        rewards = np.copy(minibatch[:, 2])
        X = np.vstack(minibatch[:, 0])
        next_state = np.vstack(minibatch[:, 3])

        if len(non_terminating[0]) > 0:
            prediction_model = self.model.predict(next_state)
            prediction_target = self.target_model.predict(next_state)
            rewards[non_terminating] += np.multiply(self.gamma,prediction_target[non_terminating, np.argmax(prediction_model[non_terminating, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(self.batch_size), actions] = rewards
        self.model.fit(X, y_target, epochs=self.epochs, verbose=self.verbose)
        self.loss.append(self.model.evaluate(X,y_target,verbose = 0))

class Agent_Linear:
    def __init__(self, nS, nA, epsilon = 1, decay = 0.99,  gamma = 0.99):
        self.nS = nS
        self.nA = nA
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = decay
        self.gamma = gamma
        self.epochs = 2
        self.verbose = 0
        self.model = self._create_model()
        self.loss = deque(maxlen=100)

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.nA, input_dim=self.nS, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=0.0001,decay=0.9,momentum=0.5))
        return model

    def action(self, curr_state):
        q = self.model.predict(curr_state)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA),q
        
        return np.argmax(q[0]),q

    def train(self,X,y):
        self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose)
        self.loss.append(self.model.evaluate(X,y,verbose = self.verbose))
        

