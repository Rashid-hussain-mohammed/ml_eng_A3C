import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import threading
import multiprocessing
import queue

# --- CONFIGURATION ---
EXE_FILENAME = "ConsumptionCar.exe"
MAX_EPISODES = 500  # Increased for better convergence
LEARNING_RATE = 0.001 # Slightly lowered for stability
GAMMA = 0.99
UPDATE_FREQ = 5 # How many steps before updating global net (n-step return)
NUM_WORKERS = multiprocessing.cpu_count() # Number of parallel threads (e.g., 4, 8, etc.)

# Global counter for episodes
global_episode_count = 0
episode_count_lock = threading.Lock()
global_rewards = []
rewards_lock = threading.Lock()

# --- ENVIRONMENT ---
class CarEnvironment:
    def __init__(self, exe_filename, worker_id=0):
        self.exe_filename = exe_filename
        self.worker_id = worker_id # Useful for debugging if needed
        
        # Check if file exists
        if not os.path.exists(self.exe_filename):
            raise FileNotFoundError(f"CRITICAL: {self.exe_filename} not found.")

        self.bounds_low = np.array([3.00, 0.40, 0.50, 0.50, 0.50])
        self.bounds_high = np.array([5.50, 0.50, 2.25, 2.25, 2.25])
        
    def step(self, action_norm):
        # 1. Denormalize actions
        real_values = self.bounds_low + action_norm * (self.bounds_high - self.bounds_low)
        
        # 2. ENFORCE CONSTRAINT: i_g3 > i_g4 > i_g5
        gears = sorted(real_values[2:], reverse=True)
        real_values[2] = gears[0]
        real_values[3] = gears[1]
        real_values[4] = gears[2]
        
        # 3. Call Simulation via WINE
        args = [str(v) for v in real_values]
        
        try:
            # Using wine
            command = ["wine", self.exe_filename] + args
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True,
                check=False 
            )
            
            output_text = result.stdout.strip()
            lines = output_text.split('\n')
            data_line = lines[-1] 
            
            outputs = [float(x) for x in data_line.replace(',', ' ').split() if x.replace('.','',1).isdigit()]
            
            if len(outputs) < 4:
                return real_values, -100.0, True # Done if error

            fc, el_g3, el_g4, el_g5 = outputs[0], outputs[1], outputs[2], outputs[3]

            # 4. Calculate Reward
            # Minimize Fuel (subtract), Maximize Elasticity (add)
            reward = (el_g3 + el_g4 + el_g5) - (fc * 2.0)

        except Exception as e:
            print(f"  [Worker {self.worker_id} Error]: {e}")
            reward = -100.0
            return real_values, reward, True
            
        return real_values, reward, False # False = Not done (continuous task, but we treat 1 step as episode for this specific problem often)

    def reset(self):
        return np.random.rand(5)

# --- NEURAL NETWORK ---
def create_model(input_shape=(5,), num_actions=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    # Actor Output
    mu = layers.Dense(num_actions, activation="sigmoid")(x) # 0-1 range
    sigma = layers.Dense(num_actions, activation="softplus")(x)
    # Critic Output
    value = layers.Dense(1)(x)
    return Model(inputs=inputs, outputs=[mu, sigma, value])

# --- A3C WORKER THREAD ---
class Worker(threading.Thread):
    def __init__(self, worker_id, global_model, optimizer):
        threading.Thread.__init__(self)
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = create_model()
        self.env = CarEnvironment(EXE_FILENAME, worker_id)
        
    def run(self):
        global global_episode_count
        
        while True:
            # Check stop condition
            with episode_count_lock:
                if global_episode_count >= MAX_EPISODES:
                    break
                current_ep = global_episode_count
                global_episode_count += 1

            # Sync local model with global weights
            self.local_model.set_weights(self.global_model.get_weights())
            
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            
            # Since this is a "One Shot" optimization (design space), 
            # we effectively have 1 step per episode. 
            # If your env was dynamic (driving a car over time), this would be a loop.
            
            with tf.GradientTape() as tape:
                # Forward pass
                mu, sigma, value = self.local_model(state)
                
                # Sample Action
                dist = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma + 1e-5)
                action = dist.sample()
                action_norm = tf.clip_by_value(action, 0.0, 1.0)
                
                # Interact with Environment
                # Note: We convert tensor to numpy for the env
                real_action = action_norm.numpy()[0]
                _, reward, done = self.env.step(real_action)
                
                # Calculate Loss
                # Since it's 1-step, Next State Value is 0 (terminal)
                # Target = Reward
                # Advantage = Reward - Value
                
                target = tf.convert_to_tensor([[reward]], dtype=tf.float32)
                advantage = target - value
                
                # Actor Loss
                log_prob = dist.log_prob(action_norm)
                actor_loss = -tf.reduce_sum(log_prob * advantage)
                
                # Critic Loss
                critic_loss = tf.square(advantage)
                
                # Entropy (optional, encourages exploration)
                entropy = dist.entropy()
                
                total_loss = tf.reduce_mean(actor_loss + critic_loss - 0.01 * entropy)

            # Calculate Gradients using LOCAL model
            grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            
            # Apply Gradients to GLOBAL model
            # This is the "Asynchronous" part
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
            
            # Logging
            with rewards_lock:
                global_rewards.append(reward)
            
            print(f"Worker {self.worker_id} | Ep {current_ep} | Reward: {reward:.4f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(EXE_FILENAME):
        print(f"ERROR: '{EXE_FILENAME}' is missing!")
        exit()

    print(f"--- Starting A3C with {NUM_WORKERS} Workers ---")
    
    # 1. Create Global Model and Optimizer
    global_model = create_model()
    # Dummy forward pass to initialize weights
    global_model(tf.convert_to_tensor(np.random.rand(1, 5), dtype=tf.float32))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # 2. Create Workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = Worker(i, global_model, optimizer)
        workers.append(worker)
        
    # 3. Start Threads
    for worker in workers:
        worker.start()
        
    # 4. Wait for completion
    for worker in workers:
        worker.join()
        
    print("\nTraining Complete.")
    
    # 5. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(global_rewards)
    plt.title("A3C Optimization Progress")
    plt.xlabel("Total Episodes (All Workers)")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("optimization_results_A3C.png")
    print("Results saved to 'optimization_results_A3C.png'")