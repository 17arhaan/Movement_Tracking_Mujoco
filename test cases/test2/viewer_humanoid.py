#!/usr/bin/env python3
from stable_baselines3 import PPO
import yaml
from data_loader import load_demonstrations
from humanoid_demo_env import HumanoidDemoEnv
import time

print("🚀 Loading trained humanoid model...")

with open("config.yaml") as f: 
    cfg = yaml.safe_load(f)

demo_states, demo_actions = load_demonstrations(cfg["csv_path"])
env = HumanoidDemoEnv(cfg["xml_path"], demo_states, demo_actions, cfg)
model = PPO.load("humanoid_trained_model", env=env)

print("✅ Model loaded successfully!")
print("🎬 Starting viewer... (Press Ctrl+C to stop)")

env.use_imitation = False
obs, _ = env.reset()

try:
    for i in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            print(f"Episode ended at step {i}, restarting...")
            obs, _ = env.reset()

        if i % 500 == 0:
            print(f"Step {i}/5000 - Reward: {reward:.3f}")

except KeyboardInterrupt:
    print("\n🛑 Viewer stopped by user")

print("✅ Viewer session completed!")
