import random
import os
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

GAMMA = 0.95


class DQLSolver:
    def __init__(
        self,
        input_shape,
        output_shape,
        learning_rate=0.001,
        batch_size=100,
        memory_size=1000000,
        EXPLORATION_MAX=1.0,
        EXPLORATION_MIN=0.01,
        EXPLORATION_DECAY=0.995,
    ):
        self.exploration_rate = EXPLORATION_MAX
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size

        self.EXPLORATION_MAX = EXPLORATION_MAX
        self.EXPLORATION_MIN = EXPLORATION_MIN
        self.EXPLORATION_DECAY = EXPLORATION_DECAY

        self.memory = deque(maxlen=memory_size)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(input_shape,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(output_shape, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

    def callback(self):
        checkpoint_path = "training/cp.pkt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            print("r")
            return random.randrange(self.output_shape)
        print("m")
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = reward + GAMMA * np.amax(self.model.predict(state_next)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0, callbacks=[self.callback()])
        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
