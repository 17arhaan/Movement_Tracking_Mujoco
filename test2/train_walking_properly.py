#!/usr/bin/env python3
"""
Enhanced Humanoid Walking Training
Fixes dimensional issues and uses proper RL training for walking.
"""

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import yaml

class HumanoidWalkingEnv(gym.Env):
    """Simplified humanoid environment focused on walking without demo data dependency."""
    
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
        
        # Walking parameters
        self.max_episode_steps = 1000
        self.step_count = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset to standing position with small random perturbations
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.3  # Set appropriate height for standing
        
        # Add small random perturbations to make training more robust
        noise_scale = 0.1
        self.data.qpos[:] += np.random.normal(0, noise_scale, size=self.model.nq)
        self.data.qpos[2] = max(1.0, self.data.qpos[2])  # Ensure minimum height
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        # Add small random velocities
        self.data.qvel[:] += np.random.normal(0, 0.1, size=self.model.nv)
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1, 1)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_walking_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """Get full observation: positions + velocities"""
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_walking_reward(self):
        """Reward function designed to encourage walking behavior"""
        
        # Get current state
        pos = self.data.qpos
        vel = self.data.qvel
        
        # Body position and orientation
        root_x, root_y, root_z = pos[0], pos[1], pos[2]
        root_vx, root_vy, root_vz = vel[0], vel[1], vel[2]
        
        # Body orientation (quaternion)
        quat = pos[3:7]
        
        # Convert quaternion to euler for easier handling
        # Simple upright check using z-axis of rotation matrix
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Reward components
        reward = 0.0
        
        # 1. Forward velocity reward (main objective)
        target_speed = 2.0  # Target walking speed
        forward_reward = 2.0 * min(root_vx / target_speed, 1.0)
        reward += forward_reward
        
        # 2. Upright posture reward
        # Reward staying upright (penalize tilting)
        upright_reward = 2.0 * (2 * (w**2 + z**2) - 1)  # Simplified upright check
        reward += upright_reward
        
        # 3. Height reward (stay off the ground)
        height_reward = 2.0 * max(0, min((root_z - 0.8) / 0.5, 1.0))
        reward += height_reward
        
        # 4. Penalty for excessive lateral movement
        lateral_penalty = -0.5 * abs(root_vy)
        reward += lateral_penalty
        
        # 5. Penalty for falling
        if root_z < 0.5:
            reward -= 10.0
        
        # 6. Action smoothness penalty
        action_penalty = -0.01 * np.sum(np.square(self.data.ctrl))
        reward += action_penalty
        
        # 7. Bonus for maintaining forward progress
        if root_vx > 0.5:
            reward += 1.0
            
        # 8. Small bonus for just surviving
        reward += 0.1
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        pos = self.data.qpos
        
        # Terminate if fallen (too low height)
        if pos[2] < 0.3:
            return True
            
        # Terminate if moved too far laterally (off track)
        if abs(pos[1]) > 5.0:
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
    """Create and wrap environment"""
    env = HumanoidWalkingEnv()
    env = Monitor(env)
    return env

def train_walking_humanoid():
    """Train humanoid to walk with proper RL approach"""
    print("🚀 Starting Enhanced Humanoid Walking Training")
    
    # Create environment
    env = DummyVecEnv([create_env])
    
    # Create PPO model with walking-focused hyperparameters
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=lambda x: x  # Linear activation for continuous control
        )
    )
    
    print("✅ Model created successfully")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Training phases
    print("🎯 Phase 1: Basic standing and balance (25k steps)")
    model.learn(total_timesteps=25000, progress_bar=True, reset_num_timesteps=True)
    model.save("humanoid_walking_25000_steps")
    
    print("🎯 Phase 2: Learning to walk (50k more steps)")
    model.learn(total_timesteps=50000, progress_bar=True, reset_num_timesteps=False)
    model.save("humanoid_walking_75000_steps")
    
    print("🎯 Phase 3: Improving walking (50k more steps)")
    model.learn(total_timesteps=50000, progress_bar=True, reset_num_timesteps=False)
    model.save("humanoid_walking_125000_steps")
    
    print("🎯 Phase 4: Final refinement (75k more steps)")
    model.learn(total_timesteps=75000, progress_bar=True, reset_num_timesteps=False)
    model.save("humanoid_walking_final")
    
    print("✅ Training completed! Total steps: 200,000")
    
    # Test the model
    print("🧪 Testing trained model...")
    test_env = create_env()
    
    obs, _ = test_env.reset()
    total_reward = 0
    episode_count = 0
    
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} completed at step {step}, reward: {total_reward:.2f}")
            obs, _ = test_env.reset()
            total_reward = 0
            
            if episode_count >= 5:  # Test 5 episodes
                break
    
    print("✅ Testing completed!")
    return model

if __name__ == "__main__":
    # Train the model
    trained_model = train_walking_humanoid()
    
    print("🎬 Training completed! To view the trained humanoid:")
    print("   mjpython viewer_walking.py") 