#!/usr/bin/env python3

import yaml
import numpy as np
from stable_baselines3 import PPO
from data_loader import load_demonstrations
from humanoid_demo_env import HumanoidDemoEnv
import mujoco
import imageio
import os

def record_humanoid_video(output_path="humanoid_walking.mp4", duration=10, fps=30):
    """Record a video of the humanoid walking"""
    
    print("🎥 Setting up video recording...")
    
    # Load config and data
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    demo_states, demo_actions = load_demonstrations(cfg["csv_path"])
    
    # Create environment
    env = HumanoidDemoEnv(
        xml_path=cfg["xml_path"],
        demo_states=demo_states,
        demo_actions=demo_actions,
        cfg=cfg
    )

    # Load model
    model = PPO.load("ppo_humanoid", env=env)
    
    # Set up rendering
    env.use_imitation = False
    obs, info = env.reset()
    
    # Calculate number of frames
    total_frames = duration * fps
    frames = []
    
    print(f"🎬 Recording {duration} seconds at {fps} FPS ({total_frames} frames)...")
    
    # Create renderer for offscreen rendering
    renderer = mujoco.Renderer(env.model, width=640, height=480)
    
    try:
        for frame_idx in range(total_frames):
            # Get action and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame
            renderer.update_scene(env.data)
            pixels = renderer.render()
            frames.append(pixels)
            
            # Reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()
            
            # Progress update
            if frame_idx % (fps * 2) == 0:  # Every 2 seconds
                print(f"📹 Recorded {frame_idx}/{total_frames} frames ({frame_idx/fps:.1f}s)")
    
        print(f"💾 Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"✅ Video saved successfully!")
        print(f"🎬 You can now play: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during recording: {e}")
    finally:
        renderer.close()

if __name__ == "__main__":
    record_humanoid_video() 