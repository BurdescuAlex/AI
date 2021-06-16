import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np

from Environment import Environment
from utility import ReplayBuffer
from DbENV import DBenv


class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(32, activation="relu")
        self.dense3 = Dense(2, dtype=tf.float32, activation="softmax")

    @tf.function
    def __call__(self, x):
        """ Apelarea retelei """
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class Agent:
    def __init__(self):
        self.main_nn = DQN()
        self.target_nn = DQN()
        self.buffer = ReplayBuffer(16000)
        self.batch_size = 100
        self.epsilon = 1.0
        self.steps_until_sync = 100,  # At how many steps should we update the target network weights
        self.optimizer = Adam(1e-4)
        self.env = Environment()
        self.dbenv = DBenv()
        self.epsilon_decay = 0.99
        self.loss_function = BinaryCrossentropy()

    def select_epsilon_greedy_action(self, state, epsilon):
        """ Ia actiune random cu probabilitatea epsilon sau alege actiunea greedy """
        result = tf.random.uniform((1,))

        if result < epsilon:
            return self.env.random_action()  # Random action
        else:
            return tf.argmax(self.main_nn(np.reshape(state, newshape=(1, len(state))))[0]).numpy()  # Greedy action.

    def train_step(self, states, actions, rewards, next_states, dones):
        """Iteratie de training pe ceea ce avem in replayBuffer."""

        with tf.GradientTape() as tape:
            dqn_variables = self.main_nn.trainable_variables
            tape.watch(dqn_variables)

            Q_nextState = self.target_nn(next_states)
            nextBestActions = tf.argmax(Q_nextState, axis=1)
            b_nextBestActions_onehotEncoding = tf.one_hot(indices=nextBestActions, depth=2)

            targetQ = b_nextBestActions_onehotEncoding * Q_nextState
            targetQ = tf.reduce_sum(targetQ, axis=1)
            targetQ = np.float32(rewards) + tf.cast((1 - dones), tf.float32) * targetQ

            Q_states = self.main_nn(states)
            b_actions_onehotEncoding = tf.one_hot(indices=actions, depth=2)
            predictedQ = b_actions_onehotEncoding * Q_states
            predictedQ = tf.reduce_sum(predictedQ, axis=1)

            loss = self.loss_function(targetQ, predictedQ)

        grads = tape.gradient(loss, dqn_variables)
        self.optimizer.apply_gradients(zip(grads, dqn_variables))
        self.epsilon *= self.epsilon_decay

        return loss

    def saveModel(self, modelName):
        self.main_nn.save_weights(modelName)

    def loadModel(self, modelName):
        self.main_nn.load_weights(modelName)
        self.target_nn.set_weights(self.main_nn.get_weights())

    def train(self):
        while True:
            state = self.env.get_state()
            ep_reward = 0
            done = 0
            weights_update = 0
            while not done:
                action = self.select_epsilon_greedy_action(state, self.epsilon)
                next_state, reward, done = self.env.step(action)
                ep_reward += reward
                # Save to experience replay.
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                weights_update += 1
                # Copy main_nn weights to target_nn.
                if weights_update % 20 == 0:
                    self.target_nn.set_weights(self.main_nn.get_weights())
                    weights_update = 0
                    self.saveModel('TrainedModel')
                print(f'Got reward: {reward}')
                # Train neural network.
                if len(self.buffer) >= self.batch_size:
                    train_batch = self.buffer.sample(self.batch_size)
                    states = tf.stack([x[0] for x in train_batch], axis=0)
                    actions = tf.stack([x[1] for x in train_batch], axis=0)
                    rewards = tf.stack([x[2] for x in train_batch], axis=0)
                    next_states = tf.stack([x[3] for x in train_batch], axis=0)
                    dones = tf.stack([x[4] for x in train_batch], axis=0)
                    loss = self.train_step(states, actions, rewards, next_states, dones)
                    print(f"Training  -> Got reward:{ep_reward}, loss ={loss}, epsilon={self.epsilon}")

    def train_db(self):
        while True:
            state = self.dbenv.state()
            ep_reward = 0
            done = 0
            weights_update = 0
            while not done:
                action = self.select_epsilon_greedy_action(state, self.epsilon)
                next_state, reward, done = self.dbenv.step(action)
                ep_reward += reward
                # Save to experience replay.
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                weights_update += 1
                # Copy main_nn weights to target_nn.
                if weights_update % 20 == 0:
                    self.target_nn.set_weights(self.main_nn.get_weights())
                    weights_update = 0
                    self.saveModel('TrainedModel')
                print(f'Got reward: {reward}')
                # Train neural network.
                if len(self.buffer) >= self.batch_size:
                    train_batch = self.buffer.sample(self.batch_size)
                    states = tf.stack([x[0] for x in train_batch], axis=0)
                    actions = tf.stack([x[1] for x in train_batch], axis=0)
                    rewards = tf.stack([x[2] for x in train_batch], axis=0)
                    next_states = tf.stack([x[3] for x in train_batch], axis=0)
                    dones = tf.stack([x[4] for x in train_batch], axis=0)
                    loss = self.train_step(states, actions, rewards, next_states, dones)
                    print(f"Training  -> Got reward:{ep_reward}, loss ={loss}, epsilon={self.epsilon}")