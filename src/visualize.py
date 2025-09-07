import time
import pygame
from utils import handle_events

# Taxi-v3 durak koordinatları (R, G, Y, B)
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# Basit duvar gösterimi (görsel amaçlı; Gym'in iç bariyerlerinin stilize hali)
WALLS = {
    (0, 0):(0, 1), (0, 1):(0, 2), (0, 4):(1, 4),
    (1, 0):(2, 0), (1, 2):(1, 3), (2, 2):(3, 2),
    (3, 0):(4, 0), (3, 3):(3, 4), (4, 1):(4, 2)
}

def init_display(config):
    """Pygame'i başlatır, ekran ve font döndürür."""
    pygame.init()
    cell = config["cell_size"]
    cols, rows = 5, 5
    width, height = cols * cell, rows * cell
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Taxi-v3 Animasyon")
    font = pygame.font.SysFont("Arial", 28, bold=True)
    return screen, font

def draw_env(screen, env, state, config):
    """Taxi-v3 haritasını çizer."""
    handle_events()

    cell = config["cell_size"]
    cols, rows = 5, 5
    width, height = cols * cell, rows * cell

    # arka plan
    screen.fill((255, 255, 255))

    # grid
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * cell, r * cell, cell, cell)
            pygame.draw.rect(screen, (200, 200, 200), rect, 0)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # duvarlar
    for (r1, c1), (r2, c2) in WALLS.items():
        x = c1 * cell
        y = r1 * cell
        if r1 == r2:  # yatay
            pygame.draw.line(screen, (0, 0, 0), (x, y), (x + cell, y), 5)
        else:         # dikey
            pygame.draw.line(screen, (0, 0, 0), (x + cell, y), (x + cell, y + cell), 5)

    # state decode
    # Gymnasium'da decode'a güvenli erişim:
    decoder = getattr(env, "decode", None)
    if decoder is None:
        # bazı sürümlerde env.decode unwrapped altında
        taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    else:
        taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)

    # duraklar
    for i, (r, c) in enumerate(LOCS):
        color = (0, 200, 0) if i == dest_idx else (160, 160, 160)
        pygame.draw.rect(screen, color, (c * cell + 20, r * cell + 20, cell - 40, cell - 40))

    # yolcu (takside değilse)
    if pass_idx < 4:
        pr, pc = LOCS[pass_idx]
        pygame.draw.circle(screen, (0, 0, 200), (pc * cell + cell // 2, pr * cell + cell // 2), 18)

    # taksi
    pygame.draw.rect(screen, (255, 200, 0), (taxi_col * cell + 10, taxi_row * cell + 10, cell - 20, cell - 20))

    pygame.display.flip()

def show_message(screen, font, text, color=(0, 0, 0), hold=2.0, config=None):
    """Ekranın ortasında yarı saydam mesaj kutusu gösterir."""
    cell = config["cell_size"] if config else 100
    cols, rows = 5, 5
    width, height = cols * cell, rows * cell

    overlay = pygame.Surface((width, height))
    overlay.set_alpha(180)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))

    msg = font.render(text, True, color)
    rect = msg.get_rect(center=(width // 2, height // 2))
    screen.blit(msg, rect)
    pygame.display.flip()
    time.sleep(hold)

def run_step(screen, font, env, state, config, delay=0.2):
    """Tek bir kare çiz ve istenen gecikme kadar bekle."""
    draw_env(screen, env, state, config)
    time.sleep(delay)

def show_message_screen(screen, font, ep, last_reward, total_reward, is_test, config):
    tag = "Test" if is_test else "Bölüm"
    if last_reward == 20:
        show_message(screen, font, f"{tag} {ep} BAŞARILI | Puan: {total_reward}", (0, 150, 0), 2.0, config)
    else:
        show_message(screen, font, f"{tag} {ep} BAŞARISIZ | Puan: {total_reward}", (200, 0, 0), 2.0, config)
