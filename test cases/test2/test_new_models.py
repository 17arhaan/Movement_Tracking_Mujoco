#!/usr/bin/env python3
"""
Test the NEW working models (no dimensional issues!)
"""

from simple_stand_walk import StandingHumanoidEnv, WalkingHumanoidEnv
from stable_baselines3 import PPO
import time

def test_standing_model():
    """Test the standing model"""
    print("🧪 Testing STANDING model...")
    
    env = StandingHumanoidEnv()
    model = PPO.load("standing_model", env=env)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if i % 50 == 0:
            pos = env.data.qpos
            print(f"Step {i:3d}: Height={pos[2]:.2f}, Reward={reward:.2f}")
        
        if terminated:
            print(f"❌ Fell at step {i}")
            break
    
    avg_reward = total_reward / (i + 1)
    final_height = env.data.qpos[2]
    
    print(f"📊 Standing Results: Avg Reward={avg_reward:.2f}, Final Height={final_height:.2f}")
    return avg_reward > 1.0

def test_walking_model():
    """Test the walking model"""
    print("\n🚶 Testing WALKING model...")
    
    env = WalkingHumanoidEnv()
    model = PPO.load("walking_model", env=env)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for i in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if i % 50 == 0:
            pos = env.data.qpos
            vel = env.data.qvel
            print(f"Step {i:3d}: Height={pos[2]:.2f}, X={pos[0]:.2f}, VelX={vel[0]:.2f}, Reward={reward:.2f}")
        
        if terminated:
            print(f"Episode ended at step {i}")
            break
    
    avg_reward = total_reward / (i + 1)
    final_pos = env.data.qpos
    distance = final_pos[0]
    
    print(f"📊 Walking Results: Avg Reward={avg_reward:.2f}, Distance={distance:.2f}m, Height={final_pos[2]:.2f}")
    return avg_reward > 1.0 and distance > 0.5

def view_walking_model():
    """View the walking model in action"""
    print("\n🎬 Viewing walking model...")
    
    env = WalkingHumanoidEnv()
    model = PPO.load("walking_model", env=env)
    
    obs, _ = env.reset()
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Render every frame
        env.render()
        time.sleep(0.02)
        
        if i % 100 == 0:
            pos = env.data.qpos
            print(f"Step {i:4d}: Height={pos[2]:.2f}, Distance={pos[0]:.2f}, Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode restarted at step {i}")
            obs, _ = env.reset()

if __name__ == "__main__":
    print("🔍 TESTING NEW WORKING MODELS")
    print("=" * 40)
    
    # Test standing
    standing_success = test_standing_model()
    
    # Test walking  
    walking_success = test_walking_model()
    
    print(f"\n📋 RESULTS:")
    print(f"Standing: {'✅ SUCCESS' if standing_success else '❌ FAILED'}")
    print(f"Walking:  {'✅ SUCCESS' if walking_success else '❌ FAILED'}")
    
    if standing_success and walking_success:
        print("\n🎉 Both models work! Starting viewer...")
        view_walking_model()
    else:
        print("\n⚠️  Models need more training") 