# 🍎 Catch Fruit With Your Head  🎮

This is a fun computer vision game where you catch AI-generated fruit images **with your head!**  
The game uses a **Variational Autoencoder (VAE)** to generate new fruit images on the fly, and **MediaPipe** for real-time head tracking. You control a green circle using your head movements via your webcam.

---

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision kagglehub torchsummary matplotlib mediapipe pygame
python game.py
```