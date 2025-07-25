#!/usr/bin/env python3
"""
Humanoid Diagnostic Script
Helps understand what's going wrong with the humanoid training.
"""

import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
import time

class DiagnosticHumanoidEnv(gym.Env):
    """Simple diagnostic environment to understand the humanoid."""
    
    def __init__(self, xml_path="humanoid.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Print model info
        print(f"🔍 HUMANOID MODEL ANALYSIS:")
        print(f"   Positions (nq): {self.model.nq}")
        print(f"   Velocities (nv): {self.model.nv}")
        print(f"   Controls (nu): {self.model.nu}")
        print(f"   Observation dim: {self.model.nq + self.model.nv}")
        
        # Observation and action spaces
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        
        self.step_count = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Start in a more natural standing position
        self.data.qpos[:] = 0
        
        # Set a reasonable standing height
        self.data.qpos[2] = 1.3  # Z position (height)
        
        # Try to set the humanoid in a more natural pose
        if self.model.nq >= 7:  # If we have quaternion
            self.data.qpos[3] = 1.0  # Quaternion w (upright)
            self.data.qpos[4:7] = 0  # Quaternion x,y,z
        
        # Initialize velocities to zero
        self.data.qvel[:] = 0
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        
        # Print initial state
        print(f"🚀 RESET - Initial position: {self.data.qpos[:7]}")
        print(f"   Height: {self.data.qpos[2]:.3f}")
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply zero actions first to see what happens
        if self.step_count < 100:
            self.data.ctrl[:] = 0  # No control - just gravity
        else:
            self.data.ctrl[:] = np.clip(action, -1, 1)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._simple_reward()
        
        # Print diagnostics every 50 steps
        if self.step_count % 50 == 0:
            pos = self.data.qpos
            vel = self.data.qvel
            print(f"Step {self.step_count:3d}: Pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"Vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) Reward={reward:.3f}")
        
        # Simple termination
        terminated = self.data.qpos[2] < 0.2  # Fallen down
        truncated = self.step_count >= 500
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _simple_reward(self):
        """Very simple reward to understand basic behavior"""
        height = self.data.qpos[2]
        
        # Just reward staying upright
        if height > 1.0:
            reward = 1.0
        elif height > 0.5:
            reward = 0.5
        else:
            reward = -1.0
            
        return reward
    
    def render(self, mode="human"):
        if self.viewer is None:
            try:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Viewer error: {e}")
                self.viewer = "disabled"
        
        if self.viewer != "disabled" and hasattr(self.viewer, 'sync'):
            self.viewer.sync()

def test_basic_physics():
    """Test basic physics without any control"""
    print("🧪 TESTING BASIC PHYSICS (No Control)")
    print("=" * 40)
    
    env = DiagnosticHumanoidEnv()
    obs, _ = env.reset()
    
    print("Testing physics for 200 steps with zero control...")
    
    for i in range(200):
        # Zero action - just let physics run
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if i % 50 == 0:
            env.render()
            time.sleep(0.1)
        
        if terminated:
            print(f"❌ Fell down at step {i}")
            break
    
    final_height = env.data.qpos[2]
    print(f"Final height: {final_height:.3f}")
    
    if final_height > 0.8:
        print("✅ Good! Humanoid can stand with zero control")
    elif final_height > 0.3:
        print("⚠️  Humanoid is crouching/sitting but not fallen")
    else:
        print("❌ Humanoid fell down - there might be model issues")

def test_random_actions():
    """Test with random actions"""
    print("\n🎲 TESTING RANDOM ACTIONS")
    print("=" * 40)
    
    env = DiagnosticHumanoidEnv()
    obs, _ = env.reset()
    
    print("Testing with random actions for 300 steps...")
    
    total_reward = 0
    
    for i in range(300):
        # Random action
        action = np.random.uniform(-0.5, 0.5, env.action_space.shape[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if i % 100 == 0:
            env.render()
            time.sleep(0.1)
        
        if terminated:
            print(f"Episode ended at step {i}")
            break
    
    avg_reward = total_reward / (i + 1)
    print(f"Average reward with random actions: {avg_reward:.3f}")

def test_existing_model():
    """Test existing trained model"""
    print("\n🤖 TESTING EXISTING MODEL")
    print("=" * 40)
    
    env = DiagnosticHumanoidEnv()
    
    # Try to load existing models
    model_files = [
        "humanoid_walking_final",
        "humanoid_pure_rl_final",
        "models/humanoid_pure_rl_50000_steps"
    ]
    
    model = None
    model_name = None
    
    for model_file in model_files:
        try:
            model = PPO.load(model_file, env=env)
            model_name = model_file
            print(f"✅ Loaded model: {model_name}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {str(e)[:50]}...")
    
    if model is None:
        print("❌ No models could be loaded")
        return
    
    obs, _ = env.reset()
    total_reward = 0
    
    print("Testing trained model for 400 steps...")
    
    for i in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if i % 100 == 0:
            env.render()
            time.sleep(0.1)
        
        if terminated:
            print(f"Model episode ended at step {i}")
            break
    
    avg_reward = total_reward / (i + 1)
    final_pos = env.data.qpos[:3]
    print(f"Model performance:")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    print(f"  Distance traveled: {final_pos[0]:.2f} meters")

def main():
    """Run all diagnostic tests"""
    print("🔍 HUMANOID DIAGNOSTIC SUITE")
    print("=" * 50)
    
    try:
        test_basic_physics()
        test_random_actions() 
        test_existing_model()
    except KeyboardInterrupt:
        print("\n⏹️  Diagnostics stopped by user")
    
    print("\n📋 DIAGNOSIS SUMMARY:")
    print("1. Check the console output above")
    print("2. If the humanoid falls immediately with zero control, the model setup has issues")
    print("3. If random actions perform better than the trained model, training failed")
    print("4. If the humanoid never moves forward, the reward function needs fixing")

if __name__ == "__main__":
    main() 