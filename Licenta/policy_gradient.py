import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from config import STATE_DIM
import numpy as np
from DbENV import DBenv


def build_policy_network():
    inputlayer = Input(shape=(STATE_DIM,))
    advantages = Input(shape=[1])
    dense1 = Dense(64, activation="relu")(inputlayer)
    dense2 = Dense(32, activation="relu")(dense1)
    dense3 = Dense(2, dtype=tf.float32, activation="softmax")(dense2)

    def custom_loss(y_true, y_pred):
        out = K.clip(y_pred, 1e-8, 1 - 1e-8)
        log_like = y_true * K.log(out)
        return K.sum(-log_like * advantages)

    policy = Model([inputlayer, advantages], [dense3])
    policy.compile(optimizer=Adam(lr=1e-4), loss=custom_loss)

    predict = Model([inputlayer], [dense3])

    return policy, predict


def learnDb(agent):
    while True:
        done = False
        state = agent.env.state()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action)
            agent.store_transition(next_state, action, reward)
            state = next_state
        agent.learn()


class Policy_gradient:

    def __init__(self):
        self.G = 0
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = build_policy_network()
        self.env = DBenv()

    def choose_action(self, observation):
        observation = np.reshape(observation, newshape=(1, len(observation)))[0]
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice([0, 1], p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), 2])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
