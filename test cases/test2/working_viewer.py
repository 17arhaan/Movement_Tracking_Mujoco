#!/usr/bin/env python3
"""
WORKING MuJoCo viewer for the new walking model
This WILL open the viewer on macOS!
"""

from simple_stand_walk import WalkingHumanoidEnv
from stable_baselines3 import PPO
import time

def main():
    print("🚀 Loading NEW walking model...")
    
    # Create environment
    env = WalkingHumanoidEnv()
    
    # Load walking model
    try:
        model = PPO.load("walking_model", env=env)
        print("✅ Walking model loaded successfully!")
    except:
        print("❌ Could not load walking_model, trying standing_model...")
        model = PPO.load("standing_model", env=env)
        print("✅ Standing model loaded!")
    
    print("🎬 Starting MuJoCo viewer...")
    print("   If viewer doesn't open automatically, check the terminal output")
    
    obs, _ = env.reset()
    episode_count = 0
    step_count = 0
    
    try:
        while True:  # Run until user stops
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Try to render
            try:
                env.render()
            except Exception as e:
                if step_count == 0:  # Only print error once
                    print(f"Viewer error: {e}")
                    print("Continuing without visual rendering...")
            
            step_count += 1
            
            # Print progress
            if step_count % 100 == 0:
                pos = env.data.qpos
                vel = env.data.qvel
                print(f"Step {step_count:4d}: Height={pos[2]:.2f}, X={pos[0]:.2f}, VelX={vel[0]:.2f}, Reward={reward:.2f}")
            
            # Reset if episode ends
            if terminated or truncated:
                episode_count += 1
                final_pos = env.data.qpos
                print(f"\n🏁 Episode {episode_count} completed:")
                print(f"   Steps: {step_count}")
                print(f"   Final height: {final_pos[2]:.2f}")
                print(f"   Distance traveled: {final_pos[0]:.2f} meters")
                print(f"   Last reward: {reward:.2f}")
                
                obs, _ = env.reset()
                step_count = 0
                
                # Small delay between episodes
                time.sleep(1.0)
            
            # Control frame rate
            time.sleep(0.02)  # 50 FPS
            
    except KeyboardInterrupt:
        print("\n🛑 Viewer stopped by user")
    
    print("✅ Viewer session completed!")

if __name__ == "__main__":
    main() 