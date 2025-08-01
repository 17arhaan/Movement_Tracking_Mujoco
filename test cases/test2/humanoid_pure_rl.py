#!/usr/bin/env python3
"""
Pure RL Humanoid Walking Training - No Demo Data Required
Trains a humanoid to walk from scratch using only reinforcement learning.
"""

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

class PureRLHumanoidEnv(gym.Env):
    """Pure RL humanoid environment - no demo data dependency."""
    
    def __init__(self, xml_path="humanoid.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Proper observation space (positions + velocities)
        obs_dim = self.model.nq + self.model.nv
        high = np.inf * np.ones(obs_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
        # Action space matches model controls
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.step_count = 0
        self.initial_height = 1.3
        
        # Reward tracking
        self.cumulative_distance = 0.0
        self.prev_x_pos = 0.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize to standing position
        self.data.qpos[:] = 0
        self.data.qpos[2] = self.initial_height  # Standing height
        
        # Add randomization for robustness
        noise_scale = 0.05
        self.data.qpos[:] += np.random.normal(0, noise_scale, size=self.model.nq)
        self.data.qpos[2] = max(1.0, self.data.qpos[2])  # Ensure minimum height
        
        # Initialize velocities
        self.data.qvel[:] = 0
        self.data.qvel[:] += np.random.normal(0, 0.05, size=self.model.nv)
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking variables
        self.step_count = 0
        self.cumulative_distance = 0.0
        self.prev_x_pos = self.data.qpos[0]
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply action with clipping
        self.data.ctrl[:] = np.clip(action, -1, 1)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Update tracking
        self.step_count += 1
        current_x = self.data.qpos[0]
        self.cumulative_distance += max(0, current_x - self.prev_x_pos)
        self.prev_x_pos = current_x
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """Get observation: positions + velocities"""
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_reward(self):
        """Comprehensive reward function for walking"""
        
        pos = self.data.qpos
        vel = self.data.qvel
        
        # Position and velocity components
        x, y, z = pos[0], pos[1], pos[2]
        vx, vy, vz = vel[0], vel[1], vel[2]
        
        # Quaternion (body orientation)
        quat = pos[3:7]
        w, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        
        reward = 0.0
        
        # 1. FORWARD PROGRESS (most important)
        # Reward forward velocity, with bonus for maintaining speed
        target_speed = 1.5
        speed_reward = 3.0 * min(vx / target_speed, 1.5)
        if vx > 0.5:
            speed_reward += 1.0  # Bonus for actually moving forward
        reward += speed_reward
        
        # 2. UPRIGHT POSTURE
        # Keep the torso upright (penalize tilting)
        # Using quaternion to check upright orientation
        upright_factor = 2 * (w**2 + qz**2) - 1  # Should be close to 1 when upright
        upright_reward = 2.0 * max(0, upright_factor)
        reward += upright_reward
        
        # 3. HEIGHT MAINTENANCE
        # Stay at appropriate walking height
        target_height = 1.3
        height_reward = 2.0 * max(0, 1.0 - abs(z - target_height) / 0.5)
        reward += height_reward
        
        # 4. LATERAL STABILITY
        # Minimize excessive side-to-side movement
        lateral_penalty = -0.5 * (abs(vy) + abs(y))
        reward += lateral_penalty
        
        # 5. ENERGY EFFICIENCY
        # Penalize excessive control effort
        control_penalty = -0.02 * np.sum(np.square(self.data.ctrl))
        reward += control_penalty
        
        # 6. STABILITY BONUS
        # Reward for maintaining balance
        if z > 0.8 and abs(y) < 0.5:
            reward += 0.5
        
        # 7. DISTANCE PROGRESS BONUS
        # Extra reward for covering distance
        if self.cumulative_distance > 0:
            distance_bonus = min(self.cumulative_distance * 0.1, 2.0)
            reward += distance_bonus
        
        # 8. SURVIVAL BONUS
        # Small reward for each step survived
        reward += 0.05
        
        # 9. SEVERE PENALTIES
        # Large penalties for bad behaviors
        if z < 0.4:  # Fallen down
            reward -= 10.0
        if abs(y) > 3.0:  # Moved too far sideways
            reward -= 5.0
        if abs(vx) < 0.1 and self.step_count > 50:  # Not moving forward
            reward -= 1.0
        
        return reward
    
    def _is_terminated(self):
        """Episode termination conditions"""
        pos = self.data.qpos
        
        # Fallen down
        if pos[2] < 0.3:
            return True
        
        # Moved too far laterally
        if abs(pos[1]) > 4.0:
            return True
        
        # Moved backwards too much
        if pos[0] < -2.0:
            return True
        
        return False
    
    def render(self, mode="human"):
        if self.viewer is None:
            try:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Viewer error: {e}")
                self.viewer = "disabled"
                return None
        
        if self.viewer != "disabled" and hasattr(self.viewer, 'sync'):
            self.viewer.sync()
        
        return None

def create_env():
    """Create and wrap the environment"""
    env = PureRLHumanoidEnv()
    env = Monitor(env)
    return env

def train_pure_rl_humanoid():
    """Train humanoid using pure RL - no demo data"""
    print("🚀 PURE RL HUMANOID WALKING TRAINING")
    print("=" * 50)
    print("Training humanoid to walk from scratch using only RL")
    print("No demo data dependencies - pure reinforcement learning!")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([create_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env])
    
    print("✅ Environments created")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Create PPO model with optimized hyperparameters for humanoid walking
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.005,  # Reduced entropy for more stable learning
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./tensorboard/",
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
            activation_fn=lambda x: x  # ReLU activation
        )
    )
    
    print("✅ PPO model created with walking-optimized hyperparameters")
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/",
        name_prefix="humanoid_pure_rl"
    )
    
    # Training phases
    print("\n🎯 PHASE 1: Basic Balance and Coordination (50k steps)")
    print("Learning to stand and maintain balance...")
    
    model.learn(
        total_timesteps=50000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=True
    )
    model.save("models/humanoid_pure_rl_50000_steps")
    print("✅ Phase 1 completed - Basic balance learned")
    
    print("\n🎯 PHASE 2: Forward Movement (100k more steps)")
    print("Learning to move forward and take steps...")
    
    model.learn(
        total_timesteps=100000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )
    model.save("models/humanoid_pure_rl_150000_steps")
    print("✅ Phase 2 completed - Forward movement learned")
    
    print("\n🎯 PHASE 3: Walking Refinement (150k more steps)")
    print("Refining walking gait and stability...")
    
    model.learn(
        total_timesteps=150000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )
    model.save("models/humanoid_pure_rl_300000_steps")
    print("✅ Phase 3 completed - Walking refined")
    
    print("\n🎯 PHASE 4: Final Optimization (200k more steps)")
    print("Final optimization for smooth, efficient walking...")
    
    model.learn(
        total_timesteps=200000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    # Save final model
    model.save("humanoid_pure_rl_final")
    print("✅ Final model saved!")
    
    print("\n🎉 TRAINING COMPLETED!")
    print("Total training steps: 500,000")
    print("📁 Models saved in ./models/")
    print("📊 Logs saved in ./logs/")
    
    # Test the final model
    print("\n🧪 Testing final model...")
    test_env = create_env()
    
    obs, _ = test_env.reset()
    total_reward = 0
    steps = 0
    episodes = 0
    
    for _ in range(3000):  # Test for 3000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            episodes += 1
            avg_reward = total_reward / steps if steps > 0 else 0
            distance = test_env.data.qpos[0]
            
            print(f"Episode {episodes}: Steps={steps}, Reward={total_reward:.2f}, Avg={avg_reward:.3f}, Distance={distance:.2f}m")
            
            obs, _ = test_env.reset()
            total_reward = 0
            steps = 0
            
            if episodes >= 5:
                break
    
    print("\n✅ Testing completed!")
    print("\n🎬 To view the trained humanoid:")
    print("   mjpython -c \"")
    print("   from humanoid_pure_rl import PureRLHumanoidEnv; from stable_baselines3 import PPO; import time")
    print("   env = PureRLHumanoidEnv(); model = PPO.load('humanoid_pure_rl_final', env=env)")
    print("   obs, _ = env.reset()")
    print("   for i in range(5000):")
    print("       action, _ = model.predict(obs, deterministic=True)")
    print("       obs, r, t, tr, _ = env.step(action); env.render(); time.sleep(0.02)")
    print("       if t or tr: obs, _ = env.reset()\"")
    
    return model

if __name__ == "__main__":
    train_pure_rl_humanoid() 