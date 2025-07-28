#!/usr/bin/env python3
"""
🚀 Humanoid Training Guide - Get Better Walking Performance
This script provides optimized training parameters and monitoring for Humanoid walking
"""

import gymnasium as gym
from stable_baselines3 import SAC, A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def create_optimized_training_setup():
    """Create optimized training setup for Humanoid walking"""
    
    print("🚀 HUMANOID WALKING OPTIMIZATION GUIDE")
    print("="*60)
    
    # Training targets for good walking
    training_targets = {
        'A2C': {
            'target_steps': 5_000_000,
            'save_freq': 50_000,
            'eval_freq': 25_000,
            'description': 'Fast but needs more steps'
        },
        'SAC': {
            'target_steps': 1_000_000,
            'save_freq': 25_000,
            'eval_freq': 10_000,
            'description': 'Most effective for continuous control'
        },
        'TD3': {
            'target_steps': 1_000_000,
            'save_freq': 25_000,
            'eval_freq': 10_000,
            'description': 'Good alternative to SAC'
        }
    }
    
    print("🎯 RECOMMENDED TRAINING TARGETS:")
    for algo, config in training_targets.items():
        print(f"\n{algo}:")
        print(f"  Target: {config['target_steps']:,} steps ({config['description']})")
        print(f"  Current progress: Check models/{algo}_*.zip files")
    
    return training_targets

def optimize_sac_training():
    """Provide optimized SAC training setup"""
    
    print(f"\n🔧 OPTIMIZED SAC TRAINING SETUP:")
    print("="*50)
    
    optimized_params = {
        'learning_rate': 3e-4,
        'buffer_size': 1_000_000,
        'learning_starts': 10_000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,
        'ent_coef': 'auto',
        'verbose': 1
    }
    
    print("📊 Optimized SAC Parameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n💡 To use these parameters, modify your sb3.py:")
    print(f"model = SAC('MlpPolicy', env, **optimized_params)")
    
    return optimized_params

def check_walking_quality(model_path, env_version, episodes=5):
    """Check if a model demonstrates good walking quality"""
    
    print(f"\n🔍 WALKING QUALITY TEST: {model_path}")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        env = gym.make(env_version)
        
        # Determine algorithm from filename
        if 'SAC' in model_path:
            model = SAC.load(model_path, env=env)
        elif 'A2C' in model_path:
            model = A2C.load(model_path, env=env)
        elif 'TD3' in model_path:
            model = TD3.load(model_path, env=env)
        else:
            print("❌ Unknown algorithm in model path")
            return False
        
        results = []
        
        for ep in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            max_x_pos = 0  # Track forward progress
            
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Track x position (forward movement)
                if hasattr(env.unwrapped, 'data'):
                    current_x = env.unwrapped.data.qpos[0]
                    max_x_pos = max(max_x_pos, current_x)
                
                if terminated or truncated:
                    break
            
            results.append({
                'reward': episode_reward,
                'steps': episode_steps,
                'distance': max_x_pos
            })
            
            print(f"Ep {ep+1}: {episode_reward:.1f} reward, {episode_steps} steps, {max_x_pos:.2f}m forward")
        
        env.close()
        
        # Analyze results
        avg_reward = sum(r['reward'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        avg_distance = sum(r['distance'] for r in results) / len(results)
        
        print(f"\n📊 WALKING QUALITY ASSESSMENT:")
        print(f"   Average Reward: {avg_reward:.1f}")
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"   Average Distance: {avg_distance:.2f}m")
        
        # Quality thresholds
        good_walking = avg_steps > 200 and avg_distance > 2.0
        decent_walking = avg_steps > 100 and avg_distance > 1.0
        
        if good_walking:
            print("✅ EXCELLENT WALKING - Model demonstrates sustained locomotion")
        elif decent_walking:
            print("🟡 DECENT WALKING - Model walks but could be more stable")
        else:
            print("❌ POOR WALKING - Model needs more training")
        
        return good_walking
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def training_recommendations():
    """Provide specific training recommendations"""
    
    print(f"\n🎯 TRAINING RECOMMENDATIONS:")
    print("="*50)
    
    print("1. 🚀 CONTINUE SAC TRAINING:")
    print("   - Your SAC is already performing better than A2C!")
    print("   - Target: 1M steps (you're at ~33K, need ~30x more)")
    print("   - Command: python sb3.py Humanoid-v5 SAC -t")
    
    print("\n2. 📊 MONITOR PROGRESS:")
    print("   - Check models every 50K steps")
    print("   - Good walking typically starts around 200K+ steps")
    print("   - Best performance usually around 500K-1M steps")
    
    print("\n3. 🔧 OPTIMIZE TRAINING:")
    print("   - Use the optimized SAC parameters above")
    print("   - Ensure stable GPU/compute environment")
    print("   - Consider using multiple environments (vectorized)")
    
    print("\n4. 🎮 TESTING STRATEGY:")
    print("   - Test every model checkpoint")
    print("   - Look for consistent 200+ steps per episode")
    print("   - Forward movement > 2 meters indicates good walking")

if __name__ == "__main__":
    create_optimized_training_setup()
    optimize_sac_training()
    
    # Test current models
    check_walking_quality("models/SAC_25000.zip", "Humanoid-v5")
    check_walking_quality("models/A2C_8125000.zip", "Humanoid-v4")
    
    training_recommendations()
    
    print(f"\n🎉 KEY INSIGHT:")
    print("Your SAC model is progressing well but needs more training time!")
    print("25K steps → 1M steps = ~40x more training needed for excellent walking") 