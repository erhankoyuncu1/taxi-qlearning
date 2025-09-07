import numpy as np
import gymnasium as gym

from config import config
from visualize import init_display
from train import train_qlearning
from test import run_test
from utils import save_qtable  # istersen eğitim sonunda kaydet

def main():
    # Ortam (Gymnasium)
    env = gym.make("Taxi-v3")

    # Görselleştirme
    screen, font = init_display(config)

    # Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)

    # Eğitim
    train_qlearning(env, config, screen, font, q_table)

    # qtable kayıt
    save_qtable(q_table, "models/qtable.npy")

    # Test
    run_test(env, config, screen, font, q_table)

if __name__ == "__main__":
    main()
