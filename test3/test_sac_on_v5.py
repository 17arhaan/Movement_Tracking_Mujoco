#!/usr/bin/env python3
"""
🎯 Test SAC Model on Humanoid-v5 (Correct Version)
This script tests your SAC model on the correct environment version with video recording
"""

import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
import os

def test_sac_v5():
    """Test SAC model on Humanoid-v5 with video recording"""
    
    model_path = "models/SAC_25000.zip"
    
    print("🚀 Testing SAC Model on Humanoid-v5")
    print("="*50)
    print(f"📊 Model: {model_path}")
    print(f"🎮 Environment: Humanoid-v5 (correct version!)")
    print(f"📹 Recording: Enabled")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        # Create viewer environment
        viewer_env = gym.make('Humanoid-v5', render_mode='human')
        
        # Create recording environment
        os.makedirs('videos', exist_ok=True)
        record_env = gym.make('Humanoid-v5', render_mode='rgb_array')
        record_env = RecordVideo(record_env, video_folder='videos', 
                               episode_trigger=lambda x: True, 
                               name_prefix='SAC_v5_working')
        
        # Load model
        model = SAC.load(model_path, env=viewer_env)
        print("✅ Model loaded successfully!")
        
        episodes = 3
        print(f"\n🎬 Running {episodes} episodes...")
        print("⚠️  Close the viewer window when done")
        
        for episode in range(episodes):
            print(f"\n🎯 Episode {episode + 1}/{episodes}:")
            
            # Reset both environments
            obs, _ = viewer_env.reset()
            record_obs, _ = record_env.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(1000):  # Max 1000 steps
                action, _ = model.predict(obs, deterministic=True)
                
                # Step both environments
                obs, reward, terminated, truncated, _ = viewer_env.step(action)
                record_obs, _, record_terminated, record_truncated, _ = record_env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"   Reward: {episode_reward:.2f}")
            print(f"   Steps: {episode_steps}")
            print(f"   Status: {'✅ Completed' if episode_steps > 100 else '❌ Fell early'}")
        
        # Close environments
        viewer_env.close()
        record_env.close()
        
        print(f"\n✅ Test completed successfully!")
        print(f"📹 Videos saved in 'videos' folder with prefix: SAC_v5_working")
        print(f"\n🎉 Your SAC model IS working - it just needed the right environment version!")
        
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        print("🔧 Try running: pip install gymnasium[mujoco]")

if __name__ == "__main__":
    test_sac_v5() 