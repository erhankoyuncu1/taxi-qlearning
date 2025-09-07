import numpy as np
import time
from visualize import run_step, show_message_screen
from utils import handle_events

def run_test(env, config, screen, font, q_table):
    test_episodes = config["test_episodes"]

    for ep in range(1, test_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        last_reward = 0
        print(f"\n[TEST] Episode {ep}")

        while not done:
            handle_events()

            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            last_reward = reward
            state = next_state

            run_step(screen, font, env, state, config, delay=0.5)

        show_message_screen(screen, font, ep, last_reward, total_reward, is_test=True, config=config)
