#!/usr/bin/env python3
"""
Viewer for the properly trained humanoid walking model.
Run with: mjpython viewer_walking.py
"""

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
import time

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

def main():
    """Load and run the trained walking model"""
    print("🚀 Loading trained humanoid walking model...")
    
    # Create environment
    env = HumanoidWalkingEnv()
    
    # Load the best available model
    model_files = [
        "humanoid_walking_final",
        "humanoid_walking_125000_steps", 
        "humanoid_walking_75000_steps",
        "humanoid_walking_25000_steps"
    ]
    
    model = None
    model_name = None
    
    for model_file in model_files:
        try:
            model = PPO.load(model_file, env=env)
            model_name = model_file
            print(f"✅ Loaded model: {model_name}")
            break
        except:
            continue
    
    if model is None:
        print("❌ No trained model found. Please run train_walking_properly.py first!")
        return
    
    print("🎬 Starting humanoid walking viewer...")
    print("   Press Ctrl+C to stop")
    print(f"   Model: {model_name}")
    
    obs, _ = env.reset()
    episode_count = 0
    total_reward = 0
    step_count = 0
    
    try:
        while True:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            time.sleep(0.02)  # 50 FPS
            
            # Print progress
            if step_count % 100 == 0:
                pos = env.data.qpos
                vel = env.data.qvel
                print(f"Step {step_count:4d} | Reward: {reward:6.3f} | Pos: ({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) | Vel: {vel[0]:5.2f} m/s")
            
            # Reset if episode ends
            if terminated or truncated:
                episode_count += 1
                avg_reward = total_reward / step_count if step_count > 0 else 0
                
                print(f"\n🏁 Episode {episode_count} completed!")
                print(f"   Steps: {step_count}")
                print(f"   Total reward: {total_reward:.2f}")
                print(f"   Average reward: {avg_reward:.3f}")
                print(f"   Distance traveled: {env.data.qpos[0]:.2f} meters")
                
                # Reset for next episode
                obs, _ = env.reset()
                total_reward = 0
                step_count = 0
                
                # Small delay between episodes
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\n🛑 Viewer stopped by user")
    
    print("✅ Viewer session completed!")

if __name__ == "__main__":
    main() 