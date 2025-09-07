import gym
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import random
import pygame
import time

# Ortam
env = gym.make("Taxi-v3")

# Q-learning parametreleri
alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 100000
show_every = 5000  # her 5000 bölümde animasyon göster

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# --- Pygame Ayarları ---
pygame.init()
cell_size = 100
cols, rows = 5, 5
width, height = cols * cell_size, rows * cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Taxi-v3 Animasyon")

font = pygame.font.SysFont("Arial", 28, bold=True)

# Durak koordinatları
locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit() 

def draw_env(state):
    handle_events() 
    screen.fill((255, 255, 255))

    # Grid çiz
    for x in range(cols):
        for y in range(rows):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 0)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # Duvarlar (Taxi-v3 sabit duvarlar)
    walls = {(0,0):(0,1), (0,1):(0,2), (0,4):(1,4),
             (1,0):(2,0), (1,2):(1,3), (2,2):(3,2),
             (3,0):(4,0), (3,3):(3,4), (4,1):(4,2)}
    for (r1,c1),(r2,c2) in walls.items():
        x = c1 * cell_size
        y = r1 * cell_size
        if r1 == r2:  # yatay duvar
            pygame.draw.line(screen, (0, 0, 0), (x, y), (x+cell_size, y), 5)
        else:  # dikey duvar
            pygame.draw.line(screen, (0, 0, 0), (x+cell_size, y), (x+cell_size, y+cell_size), 5)

    # State çözümleme
    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)

    # Durakları çiz
    for i, (r, c) in enumerate(locs):
        color = (0, 200, 0) if i == dest_idx else (150, 150, 150)
        pygame.draw.rect(screen, color,
                         (c * cell_size+20, r * cell_size+20, cell_size-40, cell_size-40))

    # Yolcuyu çiz (taksinin içinde değilse)
    if pass_idx < 4:
        pr, pc = locs[pass_idx]
        pygame.draw.circle(screen, (0, 0, 200),
                           (pc*cell_size+cell_size//2, pr*cell_size+cell_size//2), 20)

    # Taksi
    pygame.draw.rect(screen, (255, 200, 0),
                     (taxi_col * cell_size+10, taxi_row * cell_size+10,
                      cell_size-20, cell_size-20))

    pygame.display.flip()


def show_message(text, color=(0,0,0)):
    overlay = pygame.Surface((width, height))
    overlay.set_alpha(180)
    overlay.fill((255,255,255))
    screen.blit(overlay, (0,0))

    message = font.render(text, True, color)
    rect = message.get_rect(center=(width//2, height//2))
    screen.blit(message, rect)
    pygame.display.flip()
    time.sleep(2)


# --- Eğitim ---
for ep in range(1, episodes+1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        handle_events() 
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

        # Animasyon
        if ep % show_every == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            draw_env(state)
            time.sleep(0.2)

    # Episode sonunda başarı/başarısızlık mesajı
    if ep % show_every == 0:
        if reward == 20:  # taksi yolcuyu başarıyla bıraktı
            show_message(f"Bolum {ep} BAŞARILI  | Puan: {total_reward}", (0,150,0))
        else:
            show_message(f"Bolum {ep} BAŞARISIZ  | Puan: {total_reward}", (200,0,0))

print("Eğitim bitti")

# --- Test animasyonu ---
test_episodes = 3
for ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    print(f"\nTest Episode {ep+1}")
    while not done:
        handle_events() 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        draw_env(state)
        time.sleep(0.5)

    # Episode sonunda ekrana mesaj yaz
    if reward == 20:
        show_message(f"Test {ep+1} BAŞARILI  | Puan: {total_reward}", (0,150,0))
    else:
        show_message(f"Test {ep+1} BAŞARISIZ  | Puan: {total_reward}", (200,0,0))

pygame.quit()
