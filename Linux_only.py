"""
Deliverable 2 – Machine Learning Project Work
Algorithm: A3C (Asynchronous Advantage Actor-Critic)
Application: Simulation-Based Design Space Exploration (Vehicle Powertrain)

This implementation integrates the provided simulation model (ConsumptionCar.exe)
and fulfills all constraints defined in the Technical Specification (Deliverable 1).

Author: <Your Name>
"""

import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import threading
import multiprocessing
import platform

# ===================== CONFIGURATION =====================

EXE_FILENAME = "ConsumptionCar.exe"
MAX_EPISODES = 100
LEARNING_RATE = 0.001
GAMMA = 0.99
# Cap workers at 8 to prevent crashing Wine/Linux
NUM_WORKERS = min(multiprocessing.cpu_count(), 8)

# Global tracking
global_episode = 0
episode_lock = threading.Lock()
reward_history = []
reward_lock = threading.Lock()

# ===================== ENVIRONMENT =====================

class CarEnvironment:
    """
    Simulation-based environment wrapping the provided executable.
    One episode = one design evaluation (one-shot optimization).
    """

    def __init__(self, exe_filename, worker_id):
        self.exe = exe_filename
        self.worker_id = worker_id

        if not os.path.exists(self.exe):
            raise FileNotFoundError(f"{self.exe} not found")

        # Input bounds (normalized later)
        # NOTE: Tire radius bounds follow lecture interpretation (PDF contains typo)
        self.lb = np.array([3.00, 0.40, 0.50, 0.50, 0.50])
        self.ub = np.array([5.50, 0.50, 2.25, 2.25, 2.25])

    def reset(self):
        """Random normalized design vector"""
        return np.random.rand(5)

    def step(self, action_norm):
        """
        Executes the simulation for one design candidate.
        """

        # Denormalize
        x = self.lb + action_norm * (self.ub - self.lb)

        # Enforce constraint: i_g3 > i_g4 > i_g5
        gears = sorted(x[2:], reverse=True)
        x[2], x[3], x[4] = gears

        try:
            cmd = ["wine", self.exe] + [str(v) for v in x]
            result = subprocess.run(cmd, capture_output=True, text=True)

            outputs = [
                float(v) for v in result.stdout.split()
                if v.replace(".", "", 1).isdigit()
            ]

            if len(outputs) < 4:
                return x, -100.0, True

            fuel = outputs[0]
            el3, el4, el5 = outputs[1:4]

            # Reward: minimize fuel, maximize elasticity
            reward = (el3 + el4 + el5) - 2.0 * fuel

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error: {e}")
            reward = -100.0

        return x, reward, True

# ===================== MODEL =====================

def create_model():
    inputs = layers.Input(shape=(5,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)

    mu = layers.Dense(5, activation="sigmoid")(x)
    sigma = layers.Dense(5, activation="softplus")(x)
    value = layers.Dense(1)(x)

    return Model(inputs, [mu, sigma, value])

# ===================== WORKER =====================

class Worker(threading.Thread):
    def __init__(self, wid, global_model, optimizer):
        super().__init__()
        self.wid = wid
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = create_model()
        self.env = CarEnvironment(EXE_FILENAME, wid)

    def run(self):
        global global_episode

        while True:
            with episode_lock:
                if global_episode >= MAX_EPISODES:
                    break
                ep = global_episode
                global_episode += 1

            self.local_model.set_weights(self.global_model.get_weights())

            state = np.expand_dims(self.env.reset(), axis=0)

            with tf.GradientTape() as tape:
                mu, sigma, value = self.local_model(state)
                dist = tf.compat.v1.distributions.Normal(mu, sigma + 1e-5)
                action = tf.clip_by_value(dist.sample(), 0.0, 1.0)

                _, reward, _ = self.env.step(action.numpy()[0])

                target = tf.constant([[reward]], dtype=tf.float32)
                advantage = target - value

                actor_loss = -tf.reduce_sum(dist.log_prob(action) * advantage)
                critic_loss = tf.square(advantage)
                entropy = tf.reduce_sum(dist.entropy())

                loss = actor_loss + critic_loss - 0.01 * entropy

            grads = tape.gradient(loss, self.local_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))

            with reward_lock:
                reward_history.append(reward)

            print(f"Worker {self.wid} | Episode {ep} | Reward {reward:.2f}")

# ===================== MAIN =====================

if __name__ == "__main__":

    print(f"Starting A3C with {NUM_WORKERS} workers")

    global_model = create_model()
    global_model(np.random.rand(1, 5))

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    workers = [Worker(i, global_model, optimizer) for i in range(NUM_WORKERS)]

    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("Training complete.")

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.xlabel("Total Episodes")
    plt.ylabel("Reward")
    plt.title("A3C Optimization Progress")
    plt.grid(True)
    plt.savefig("optimization_results_A3C.png")
