import numpy as np
import random
from visualize import run_step, show_message_screen
from utils import handle_events

def train_qlearning(env, config, screen, font, q_table):
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    episodes = config["episodes"]
    show_every = config["show_every"]

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        last_reward = 0

        while not done:
            handle_events()

            # e-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            # Gymnasium step: (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            last_reward = reward

            # Q-güncelle
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state

            # animasyon: belirli aralıklarla göster
            if ep % show_every == 0:
                run_step(screen, font, env, state, config, delay=0.2)

        if ep % show_every == 0:
            show_message_screen(screen, font, ep, last_reward, total_reward, is_test=False, config=config)

    print("[INFO] Eğitim bitti")
