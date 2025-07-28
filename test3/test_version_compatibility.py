#!/usr/bin/env python3
"""
🔬 Humanoid Version Compatibility Test
Tests your SAC model on both Humanoid-v4 and Humanoid-v5 to show the difference
"""

import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

def test_model_on_version(model_path, version, episodes=3):
    """Test model on specific Humanoid version"""
    print(f"\n🧪 Testing {model_path} on {version}")
    print("="*50)
    
    try:
        # Create environment
        env = gym.make(version, render_mode='human')
        
        # Load model
        model = SAC.load(model_path, env=env)
        
        total_rewards = []
        total_steps = []
        
        for ep in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            print(f"Episode {ep+1}: ", end="", flush=True)
            
            for step in range(1000):  # Max 1000 steps
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            print(f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        env.close()
        
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        
        print(f"\n📊 {version} Results:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"   Success Rate: {(np.array(total_steps) > 100).mean()*100:.1f}%")
        
        return avg_reward, avg_steps
        
    except Exception as e:
        print(f"❌ Error testing {version}: {e}")
        return 0, 0

def compare_environments():
    """Compare observation spaces between versions"""
    print("\n🔍 Environment Comparison:")
    print("="*50)
    
    v4_env = gym.make('Humanoid-v4')
    v5_env = gym.make('Humanoid-v5')
    
    print(f"Humanoid-v4 Observation Space: {v4_env.observation_space.shape}")
    print(f"Humanoid-v5 Observation Space: {v5_env.observation_space.shape}")
    print(f"Action Space (both): {v4_env.action_space.shape}")
    
    v4_env.close()
    v5_env.close()

if __name__ == "__main__":
    print("🚀 Humanoid Version Compatibility Test")
    print("This script will show you why your SAC model behaves differently on v4 vs v5")
    
    # Compare environments first
    compare_environments()
    
    # Test your SAC model on both versions
    model_path = "models/SAC_25000.zip"  # Your latest SAC model
    
    print(f"\n🎯 Testing model: {model_path}")
    print("(Close viewer windows when each test completes)")
    
    # Test on v4 (what it was trained on)
    v4_reward, v4_steps = test_model_on_version(model_path, "Humanoid-v4", episodes=2)
    
    # Test on v5 (what you're trying now)
    v5_reward, v5_steps = test_model_on_version(model_path, "Humanoid-v5", episodes=2)
    
    # Summary
    print("\n🏁 SUMMARY")
    print("="*50)
    print(f"Humanoid-v4: Avg Reward = {v4_reward:.2f}, Avg Steps = {v4_steps:.1f}")
    print(f"Humanoid-v5: Avg Reward = {v5_reward:.2f}, Avg Steps = {v5_steps:.1f}")
    print(f"Performance Drop: {((v4_reward - v5_reward) / max(v4_reward, 1) * 100):.1f}%")
    
    print("\n💡 RECOMMENDATIONS:")
    if v4_reward > v5_reward * 1.5:
        print("✅ Your model works much better on v4!")
        print("🎯 Either:")
        print("   1. Use Humanoid-v4 for testing existing models")
        print("   2. Retrain your model specifically on Humanoid-v5")
    else:
        print("🤔 Model performs similarly on both versions")
        print("   The issue might be elsewhere") 