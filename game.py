import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pygame
import threading
import random
import time

from model import VAE

# --- HYPERPARAMETERS ---
IMAGE_SIZE = 64
LATENT_DIM = 40
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR_RATE = 1e-3
BETA = 1

# ---------- CONFIG ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WIDTH, HEIGHT = 1280, 720
FRUIT_FOLDER = "random_fruit"
HEAD_RADIUS = 80

# ---------- LOAD VAE MODEL ----------
weights_path = 'saved_weights_VAE/vae_weights20250507_232504.pth'
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

# ---------- GLOBALS ----------
head_x, head_y = WIDTH // 2, HEIGHT + 500
lock = threading.Lock()

# ---------- HEAD TRACKER ----------
def run_head_tracker():
    global head_x, head_y
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                with lock:
                    head_x = WIDTH - int(nose.x * WIDTH)
                    head_y = int(nose.y * HEIGHT)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

# ---------- VAE IMAGE GENERATOR THREAD ----------
def generate_vae_images():
    os.makedirs(FRUIT_FOLDER, exist_ok=True)
    while True:
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            generated = model.decode(z)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(FRUIT_FOLDER, f'fruit_{timestamp}.png')
        save_image(generated, filename)
        print(f"Saved VAE image: {filename}")
        time.sleep(5)  # generate every 5 seconds

# ---------- PYGAME SETUP ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch Fruit With Your Head")
clock = pygame.time.Clock()

def load_fruit_images():
    images = []
    for file in os.listdir(FRUIT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img = pygame.image.load(os.path.join(FRUIT_FOLDER, file))
            img = pygame.transform.scale(img, (100, 90))
            images.append(img)
    return images

class FallingFruit:
    def __init__(self, image):
        self.image = image
        self.x = random.randint(0, WIDTH - 100)
        self.y = random.randint(-HEIGHT, 0)
        self.speed = random.uniform(4, 8)
        self.sliced = False

    def update(self):
        if not self.sliced:
            self.y += self.speed

    def draw(self, surface):
        if not self.sliced:
            surface.blit(self.image, (self.x, self.y))

    def check_slice(self, hx, hy):
        if not self.sliced and abs(self.x - hx) < HEAD_RADIUS and abs(self.y - hy) < HEAD_RADIUS:
            self.sliced = True
            return True
        return False
    
def spawn_fruit_wave(images, count=10):
    return [FallingFruit(random.choice(images)) for _ in range(count)]

# ---------- START THREADS ----------
threading.Thread(target=run_head_tracker, daemon=True).start()
threading.Thread(target=generate_vae_images, daemon=True).start()

# ---------- GAME LOOP ----------
score = 0
game_duration = 60
start_time = time.time()
fruits = []
running = True

while running:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with lock:
        hx, hy = head_x, head_y

    if not fruits or all(f.y > HEIGHT or f.sliced for f in fruits):
        fruit_images = load_fruit_images()
        if fruit_images:
            fruits = spawn_fruit_wave(fruit_images)

    for fruit in fruits:
        fruit.update()
        if fruit.check_slice(hx, hy):
            score += 1
        fruit.draw(screen)

    pygame.draw.circle(screen, (0, 255, 0), (hx, hy), HEAD_RADIUS, 2)

    font = pygame.font.SysFont(None, 60)
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (50, 50))

    elapsed_time = time.time() - start_time
    remaining_time = max(0, int(game_duration - elapsed_time))
    timer_text = font.render(f"Time: {remaining_time}", True, (0, 0, 0))
    screen.blit(timer_text, (WIDTH - 200, 50))

    if remaining_time <= 0:
        running = False

    pygame.display.flip()
    clock.tick(60)

# ---------- GAME OVER ----------
screen.fill((255, 255, 255))
font_large = pygame.font.SysFont(None, 100)
font_small = pygame.font.SysFont(None, 60)
game_over_text = font_large.render("Game Over!", True, (255, 0, 0))
final_score_text = font_small.render(f"Your Score: {score}", True, (0, 0, 0))
screen.blit(game_over_text, (WIDTH // 2 - 200, HEIGHT // 2 - 100))
screen.blit(final_score_text, (WIDTH // 2 - 150, HEIGHT // 2))
pygame.display.flip()
pygame.time.wait(5000)
pygame.quit()
