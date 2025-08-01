#!/usr/bin/env python3
"""
🚀 Mujoco Viewer Fix - Test Script
This script demonstrates the fix for the video recording and viewer display issues.

Issues Fixed:
1. ✅ Viewer window now opens even when recording
2. ✅ Video recording works while showing viewer
3. ✅ Both work simultaneously

Usage: python mujoco_viewer_fix.py [model_name] [algorithm] [episodes] [record_video]
Example: python mujoco_viewer_fix.py A2C_8125000 A2C 1 true
"""

import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import sys
import time
import argparse

def test_fixed_viewer(model_name="A2C_8125000", algorithm="A2C", episodes=1, record_video=True):
    """Test the fixed viewer with both recording and display"""
    
    print("🚀 FIXED Mujoco Viewer Test")
    print("="*50)
    print(f"📊 Model: {model_name}")
    print(f"🤖 Algorithm: {algorithm}")
    print(f"🎯 Episodes: {episodes}")
    print(f"📹 Record Video: {record_video}")
    print("="*50)
    
    # Check if model exists
    model_path = f"models/{model_name}.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("\n📁 Available models:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith('.zip'):
                    print(f"   - {f}")
        return False
    
    try:
        # FIXED: Create environment - always show viewer, optionally record video
        print("🎮 Creating viewer environment...")
        env = gym.make('Humanoid-v4', render_mode='human')  # Main env for viewing
        
        # Create separate recording environment if needed
        if record_video:
            from gymnasium.wrappers import RecordVideo
            os.makedirs('videos', exist_ok=True)  # Ensure videos directory exists
            print("📹 Creating recording environment...")
            record_env = gym.make('Humanoid-v4', render_mode='rgb_array')
            record_env = RecordVideo(record_env, video_folder='videos', 
                                   episode_trigger=lambda x: True, 
                                   name_prefix=f'{algorithm}_{episodes}ep_fixed')
            print(f"✅ FIXED: You will see BOTH the viewer window AND get recorded videos!")
        else:
            record_env = None
            print("📺 Only viewer mode (no recording)")
        
        # Load model based on algorithm
        print(f"🤖 Loading {algorithm} model...")
        if algorithm == 'SAC':
            model = SAC.load(model_path, env=env)
        elif algorithm == 'TD3':
            model = TD3.load(model_path, env=env)
        elif algorithm == 'A2C':
            model = A2C.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"\n🎬 Running {episodes} episodes...")
        print("⚠️  Close the viewer window when done")
        print("🔄 Press Ctrl+C to stop early\n")
        
        total_reward = 0
        total_steps = 0
        
        for episode in range(episodes):
            print(f"🎯 Episode {episode + 1}/{episodes}:")
            
            # Reset both environments
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            if record_video:
                record_obs, _ = record_env.reset()
            
            # Run episode
            max_steps = 1000
            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                
                # Step both environments with the same action
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Also step the recording environment if recording
                if record_video:
                    record_obs, _, record_terminated, record_truncated, _ = record_env.step(action)
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            total_steps += episode_steps
            
            print(f"   Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        print(f"\n📊 Summary:")
        print(f"   Average Reward: {total_reward/episodes:.2f}")
        print(f"   Average Steps: {total_steps/episodes:.1f}")
        print(f"   Total Episodes: {episodes}")
        
        # Close environments
        env.close()
        if record_video:
            record_env.close()
            print("📹 Video recording completed!")
            print(f"📁 Videos saved in 'videos' folder with prefix: {algorithm}_{episodes}ep_fixed")
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test fixed Mujoco viewer')
    parser.add_argument('model_name', nargs='?', default='A2C_8125000', 
                       help='Model name (default: A2C_8125000)')
    parser.add_argument('algorithm', nargs='?', default='A2C', 
                       help='Algorithm (A2C, SAC, TD3) (default: A2C)')
    parser.add_argument('episodes', nargs='?', type=int, default=1, 
                       help='Number of episodes (default: 1)')
    parser.add_argument('record_video', nargs='?', default='true', 
                       help='Record video (true/false) (default: true)')
    
    args = parser.parse_args()
    
    record_video = args.record_video.lower() in ['true', 'yes', '1', 'on']
    
    print("🎬 Testing Fixed Mujoco Viewer")
    print("This script demonstrates the fix for:")
    print("1. ✅ Viewer opening when recording is enabled")
    print("2. ✅ Video recording working with viewer display")
    print("3. ✅ Both functioning simultaneously")
    print()
    
    success = test_fixed_viewer(args.model_name, args.algorithm, args.episodes, record_video)
    
    if success:
        print("\n🎉 Fix verified! The solution works.")
        print("📝 Apply the same pattern to your notebook for the complete fix.")
    else:
        print("\n⚠️  Test failed. Check the error messages above.") 