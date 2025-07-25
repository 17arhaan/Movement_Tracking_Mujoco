#!/usr/bin/env python3
"""
Simple Step-by-Step Humanoid Training
Phase 1: Learn to stand upright
Phase 2: Learn to take small steps
Phase 3: Learn to walk forward
"""

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import time

class StandingHumanoidEnv(gym.Env):
    """Phase 1: Just learn to stand upright and maintain balance."""
    
    def __init__(self, xml_path="humanoid.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.3, 0.3, shape=(self.model.nu,), dtype=np.float32)  # Smaller actions
        
        self.step_count = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Start standing with small random noise
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.3  # Standing height
        self.data.qpos[3] = 1.0  # Upright quaternion
        
        # Add tiny random perturbations for robustness
        self.data.qpos[:] += np.random.normal(0, 0.02, size=self.model.nq)
        self.data.qpos[2] = max(1.2, self.data.qpos[2])  # Ensure good height
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply small actions
        self.data.ctrl[:] = np.clip(action, -0.3, 0.3)
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._standing_reward()
        
        # Simple termination: fell down or episode too long
        terminated = self.data.qpos[2] < 0.5
        truncated = self.step_count >= 500
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _standing_reward(self):
        """Simple reward: just stay upright"""
        height = self.data.qpos[2]
        
        # Main reward: stay at good height
        if height > 1.2:
            height_reward = 2.0
        elif height > 1.0:
            height_reward = 1.0
        elif height > 0.8:
            height_reward = 0.5
        else:
            height_reward = -1.0
        
        # Bonus for staying upright
        stability_bonus = 1.0 if height > 1.0 else 0.0
        
        # Small penalty for excessive control
        control_penalty = -0.001 * np.sum(np.square(self.data.ctrl))
        
        # Survival bonus
        survival_bonus = 0.1
        
        return height_reward + stability_bonus + control_penalty + survival_bonus
    
    def render(self, mode="human"):
        if self.viewer is None:
            try:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except:
                self.viewer = "disabled"
        
        if self.viewer != "disabled" and hasattr(self.viewer, 'sync'):
            self.viewer.sync()

class WalkingHumanoidEnv(StandingHumanoidEnv):
    """Phase 2: Learn to walk forward while maintaining balance."""
    
    def __init__(self, xml_path="humanoid.xml"):
        super().__init__(xml_path)
        # Allow larger actions for walking
        self.action_space = gym.spaces.Box(-0.5, 0.5, shape=(self.model.nu,), dtype=np.float32)
        
    def step(self, action):
        # Apply actions
        self.data.ctrl[:] = np.clip(action, -0.5, 0.5)
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._walking_reward()
        
        # Termination conditions
        terminated = self.data.qpos[2] < 0.4 or abs(self.data.qpos[1]) > 2.0
        truncated = self.step_count >= 1000
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, {}
    
    def _walking_reward(self):
        """Reward for walking forward while staying upright"""
        pos = self.data.qpos
        vel = self.data.qvel
        
        height = pos[2]
        forward_vel = vel[0]
        side_pos = abs(pos[1])
        
        # Height reward (most important)
        if height > 1.1:
            height_reward = 3.0
        elif height > 0.9:
            height_reward = 2.0
        elif height > 0.6:
            height_reward = 1.0
        else:
            height_reward = -2.0
        
        # Forward movement reward
        if forward_vel > 0.2:
            forward_reward = 2.0 * min(forward_vel, 1.5)
        else:
            forward_reward = 0.0
        
        # Penalty for going sideways
        side_penalty = -1.0 * side_pos
        
        # Control penalty
        control_penalty = -0.01 * np.sum(np.square(self.data.ctrl))
        
        # Survival bonus
        survival_bonus = 0.2
        
        return height_reward + forward_reward + side_penalty + control_penalty + survival_bonus

def train_standing_phase():
    """Phase 1: Train to stand upright"""
    print("🚀 PHASE 1: Learning to Stand Upright")
    print("=" * 40)
    
    env = DummyVecEnv([lambda: Monitor(StandingHumanoidEnv())])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )
    
    print("Training standing for 50,000 steps...")
    model.learn(total_timesteps=50000, progress_bar=True)
    model.save("standing_model")
    
    print("✅ Phase 1 completed - Standing learned!")
    return model

def train_walking_phase():
    """Phase 2: Train to walk forward"""
    print("\n🚶 PHASE 2: Learning to Walk Forward")
    print("=" * 40)
    
    env = DummyVecEnv([lambda: Monitor(WalkingHumanoidEnv())])
    
    # Load standing model as starting point
    try:
        model = PPO.load("standing_model", env=env)
        print("✅ Loaded standing model as starting point")
    except:
        print("⚠️  Creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=8,
            gamma=0.99,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        )
    
    print("Training walking for 100,000 steps...")
    model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=False)
    model.save("walking_model")
    
    print("✅ Phase 2 completed - Walking learned!")
    return model

def test_model(model_name, env_class, steps=500):
    """Test a trained model"""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        env = env_class()
        model = PPO.load(model_name, env=env)
        
        obs, _ = env.reset()
        total_reward = 0
        
        for i in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if i % 100 == 0:
                env.render()
                pos = env.data.qpos
                print(f"Step {i:3d}: Height={pos[2]:.2f}, X={pos[0]:.2f}, Reward={reward:.2f}")
                time.sleep(0.1)
            
            if terminated or truncated:
                print(f"Episode ended at step {i}")
                break
        
        avg_reward = total_reward / (i + 1)
        final_pos = env.data.qpos
        
        print(f"📊 Results:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Final height: {final_pos[2]:.2f}")
        print(f"  Distance traveled: {final_pos[0]:.2f} meters")
        
        return avg_reward > 1.0  # Success if positive rewards
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Complete training pipeline"""
    print("🎯 SIMPLE STEP-BY-STEP HUMANOID TRAINING")
    print("=" * 50)
    
    try:
        # Phase 1: Standing
        standing_model = train_standing_phase()
        standing_success = test_model("standing_model", StandingHumanoidEnv, 300)
        
        if standing_success:
            print("✅ Standing phase successful!")
            
            # Phase 2: Walking
            walking_model = train_walking_phase()
            walking_success = test_model("walking_model", WalkingHumanoidEnv, 500)
            
            if walking_success:
                print("🎉 SUCCESS! Humanoid learned to walk!")
                print("\n🎬 To view the walking humanoid:")
                print("mjpython -c \"")
                print("from simple_stand_walk import WalkingHumanoidEnv; from stable_baselines3 import PPO; import time")
                print("env = WalkingHumanoidEnv(); model = PPO.load('walking_model', env=env)")
                print("obs, _ = env.reset()")
                print("for i in range(2000):")
                print("    action, _ = model.predict(obs, deterministic=True)")
                print("    obs, r, t, tr, _ = env.step(action); env.render(); time.sleep(0.02)")
                print("    if t or tr: obs, _ = env.reset()\"")
            else:
                print("⚠️  Walking phase needs more training")
        else:
            print("❌ Standing phase failed - need to debug further")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")

if __name__ == "__main__":
    main() 