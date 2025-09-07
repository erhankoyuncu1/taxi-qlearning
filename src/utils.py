import os
import numpy as np
import pygame

def handle_events():
    """Pencere kapatma vb. olayları ele alır. Kapatılırsa programdan çıkar."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

def save_qtable(q_table, path="models/qtable.npy"):
    """Q-table'ı .npy dosyasına kaydeder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, q_table)
    print(f"[INFO] Q-table kaydedildi: {path}")

def load_qtable(path="models/qtable.npy"):
    """Kaydedilmiş Q-table'ı yükler."""
    if os.path.exists(path):
        q_table = np.load(path)
        print(f"[INFO] Q-table yüklendi: {path}")
        return q_table
    raise FileNotFoundError(f"Q-table dosyası bulunamadı: {path}")
