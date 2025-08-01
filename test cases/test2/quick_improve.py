#!/usr/bin/env python3
"""
Quick Walking Improvement
Just train the existing model more with better settings
"""

from simple_stand_walk import WalkingHumanoidEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import time

def quick_improve():
    """Quick improvement of existing walking model"""
    print("🚀 QUICK WALKING IMPROVEMENT")
    print("=" * 40)
    
    # Create environment
    env = DummyVecEnv([lambda: Monitor(WalkingHumanoidEnv())])
    
    # Load existing model
    try:
        model = PPO.load("walking_model", env=env)
        print("✅ Loaded existing walking model")
    except:
        print("❌ No walking model found. Run simple_stand_walk.py first!")
        return
    
    # Lower learning rate for fine-tuning
    model.learning_rate = 1e-4
    print(f"🎯 Fine-tuning with learning rate: {model.learning_rate}")
    
    # Additional training with tensorboard
    print("🔥 Additional training for 150,000 steps...")
    model.learn(
        total_timesteps=150000,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name="quick_improve"
    )
    
    # Save improved model
    model.save("improved_walking_model")
    print("✅ Quick improvement completed!")
    
    # Test the improved model
    print("\n🧪 Testing improved model...")
    test_env = WalkingHumanoidEnv()
    obs, _ = test_env.reset()
    
    total_reward = 0
    max_distance = 0
    episode_count = 0
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        
        if i % 200 == 0:
            pos = test_env.data.qpos
            distance = pos[0]
            max_distance = max(max_distance, distance)
            print(f"Step {i:3d}: Height={pos[2]:.2f}, Distance={distance:.2f}, Reward={reward:.2f}")
        
        if terminated or truncated:
            episode_count += 1
            final_pos = test_env.data.qpos
            print(f"🏁 Episode {episode_count}: Distance={final_pos[0]:.2f}m, Reward={total_reward/i:.2f}")
            obs, _ = test_env.reset()
            total_reward = 0
            
            if episode_count >= 3:
                break
    
    print(f"\n📊 IMPROVEMENT RESULTS:")
    print(f"  Max distance: {max_distance:.2f} meters")
    print(f"  Episodes completed: {episode_count}")
    
    return model

def view_improved():
    """View the improved walking model"""
    print("🎬 Viewing improved walking model...")
    
    try:
        env = WalkingHumanoidEnv()
        model = PPO.load("improved_walking_model", env=env)
        print("✅ Loaded improved model")
    except:
        print("❌ No improved model found. Run quick_improve() first!")
        return
    
    obs, _ = env.reset()
    
    for i in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        env.render()
        time.sleep(0.02)
        
        if i % 100 == 0:
            pos = env.data.qpos
            vel = env.data.qvel
            print(f"Step {i:4d}: Height={pos[2]:.2f}, Distance={pos[0]:.2f}, VelX={vel[0]:.2f}, Reward={reward:.2f}")
        
        if terminated or truncated:
            final_pos = env.data.qpos
            print(f"🏁 Episode ended: Distance={final_pos[0]:.2f}m")
            obs, _ = env.reset()

if __name__ == "__main__":
    # Quick improvement
    model = quick_improve()
    
    print("\n🎬 To view the improved model:")
    print("mjpython quick_improve.py  # Then call view_improved()")
    print("\n📊 To view training progress:")
    print("tensorboard --logdir=./logs/") 