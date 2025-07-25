#!/usr/bin/env python3
"""
Quick test to verify the standing approach works
"""

import numpy as np
from simple_stand_walk import StandingHumanoidEnv
import time

def test_standing_manually():
    """Test if a simple PD controller can make the humanoid stand"""
    print("🧪 QUICK STANDING TEST")
    print("Testing if simple control can make humanoid stand...")
    
    env = StandingHumanoidEnv()
    obs, _ = env.reset()
    
    total_reward = 0
    
    for i in range(300):
        # Simple PD controller to maintain standing pose
        pos = env.data.qpos
        vel = env.data.qvel
        
        # Target: stay at height 1.3, minimal tilt
        target_height = 1.3
        height_error = target_height - pos[2]
        
        # Simple control: small actions to correct height and balance
        action = np.zeros(env.action_space.shape[0])
        
        # If falling, apply upward torque to key joints
        if height_error > 0.1:  # Too low
            action[6:12] = 0.1  # Leg joints - small upward force
        elif height_error < -0.1:  # Too high
            action[6:12] = -0.05  # Slight downward
        
        # Balance correction
        if abs(pos[1]) > 0.1:  # Tilting sideways
            action[3:6] = -pos[1] * 0.2  # Counter-tilt
        
        # Apply action
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if i % 50 == 0:
            env.render()
            print(f"Step {i:3d}: Height={pos[2]:.2f}, Reward={reward:.2f}, Total={total_reward:.1f}")
            time.sleep(0.1)
        
        if terminated:
            print(f"❌ Fell down at step {i}")
            break
    
    avg_reward = total_reward / (i + 1)
    final_height = env.data.qpos[2]
    
    print(f"\n📊 RESULTS:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Final height: {final_height:.2f}")
    print(f"  Success: {'✅ YES' if avg_reward > 1.0 and final_height > 0.8 else '❌ NO'}")
    
    if avg_reward > 1.0:
        print("🎉 Standing works! The training should succeed!")
    else:
        print("⚠️  Need to adjust the approach")

def test_random_vs_zero():
    """Compare random actions vs zero actions"""
    print("\n🎲 RANDOM vs ZERO CONTROL TEST")
    
    # Test 1: Zero control
    print("Testing zero control...")
    env = StandingHumanoidEnv()
    obs, _ = env.reset()
    
    zero_reward = 0
    for i in range(100):
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        zero_reward += reward
        if terminated:
            break
    zero_avg = zero_reward / (i + 1)
    zero_height = env.data.qpos[2]
    
    # Test 2: Small random control
    print("Testing small random control...")
    env = StandingHumanoidEnv()
    obs, _ = env.reset()
    
    random_reward = 0
    for i in range(100):
        action = np.random.uniform(-0.1, 0.1, env.action_space.shape[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        random_reward += reward
        if terminated:
            break
    random_avg = random_reward / (i + 1)
    random_height = env.data.qpos[2]
    
    print(f"\n📊 COMPARISON:")
    print(f"  Zero control:   Reward={zero_avg:.2f}, Height={zero_height:.2f}")
    print(f"  Random control: Reward={random_avg:.2f}, Height={random_height:.2f}")
    
    if zero_avg > 1.5:
        print("✅ Zero control works well - RL should learn easily!")
    elif random_avg > zero_avg:
        print("⚠️  Random control better - need more exploration")
    else:
        print("🤔 Both struggle - may need environment tweaks")

if __name__ == "__main__":
    test_standing_manually()
    test_random_vs_zero() 