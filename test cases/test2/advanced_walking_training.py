#!/usr/bin/env python3
"""
Advanced Walking Training with Multiple Improvements
- Tensorboard monitoring
- Better reward function
- Curriculum learning
- Hyperparameter optimization
"""

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import time
import os

class ImprovedWalkingEnv(gym.Env):
    """Enhanced walking environment with better reward shaping"""
    
    def __init__(self, xml_path="humanoid.xml", difficulty=1.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Difficulty scaling (0.5 = easier, 2.0 = harder)
        self.difficulty = difficulty
        
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.8, 0.8, shape=(self.model.nu,), dtype=np.float32)
        
        # Episode parameters
        self.max_episode_steps = int(1500 / difficulty)  # Easier = longer episodes
        self.step_count = 0
        
        # Reward tracking
        self.prev_x_pos = 0.0
        self.cumulative_distance = 0.0
        self.stability_bonus = 0.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset to standing with slight randomization
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.3  # Standing height
        self.data.qpos[3] = 1.0  # Upright quaternion
        
        # Add small perturbations for robustness
        noise_scale = 0.03 / self.difficulty  # Easier = less noise
        self.data.qpos[:] += np.random.normal(0, noise_scale, size=self.model.nq)
        self.data.qpos[2] = max(1.1, self.data.qpos[2])
        
        # Zero velocities with small random component
        self.data.qvel[:] = 0
        self.data.qvel[:] += np.random.normal(0, 0.02, size=self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking
        self.step_count = 0
        self.prev_x_pos = self.data.qpos[0]
        self.cumulative_distance = 0.0
        self.stability_bonus = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply action with scaling
        action_scale = 0.7 / self.difficulty  # Easier = smaller actions
        self.data.ctrl[:] = np.clip(action * action_scale, -0.8, 0.8)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_enhanced_reward()
        
        # Enhanced termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        self.step_count += 1
        
        # Update tracking
        current_x = self.data.qpos[0]
        forward_progress = max(0, current_x - self.prev_x_pos)
        self.cumulative_distance += forward_progress
        self.prev_x_pos = current_x
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _compute_enhanced_reward(self):
        """Enhanced reward function with better walking incentives"""
        pos = self.data.qpos
        vel = self.data.qvel
        
        # Position and velocity components
        x, y, z = pos[0], pos[1], pos[2]
        vx, vy, vz = vel[0], vel[1], vel[2]
        
        # Quaternion for orientation
        quat = pos[3:7]
        w, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        
        reward = 0.0
        
        # 1. FORWARD VELOCITY (primary objective)
        target_speed = 1.2 * self.difficulty
        if vx > 0.3:
            forward_reward = 4.0 * min(vx / target_speed, 1.5)
            # Bonus for maintaining good speed
            if 0.8 <= vx <= 2.0:
                forward_reward += 2.0
        else:
            forward_reward = -0.5  # Penalty for not moving forward
        reward += forward_reward
        
        # 2. HEIGHT STABILITY (crucial for walking)
        target_height = 1.3
        height_diff = abs(z - target_height)
        if height_diff < 0.1:
            height_reward = 3.0
        elif height_diff < 0.3:
            height_reward = 2.0 * (1.0 - height_diff / 0.3)
        else:
            height_reward = -2.0
        reward += height_reward
        
        # 3. UPRIGHT ORIENTATION
        upright_factor = 2 * (w**2 + qz**2) - 1
        upright_reward = 2.0 * max(0, upright_factor)
        reward += upright_reward
        
        # 4. LATERAL STABILITY
        lateral_penalty = -2.0 * (abs(vy) + abs(y))
        reward += lateral_penalty
        
        # 5. GAIT SMOOTHNESS (encourage natural walking)
        action_smoothness = -0.01 * np.sum(np.square(self.data.ctrl))
        reward += action_smoothness
        
        # 6. PROGRESS BONUS
        if self.cumulative_distance > 0:
            progress_bonus = min(self.cumulative_distance * 0.5, 3.0)
            reward += progress_bonus
        
        # 7. STABILITY BONUS (for staying upright over time)
        if z > 1.0 and abs(y) < 0.3:
            self.stability_bonus += 0.01
            reward += min(self.stability_bonus, 1.0)
        
        # 8. STEP SURVIVAL BONUS
        reward += 0.05
        
        # 9. PENALTY SYSTEM
        if z < 0.5:  # Fallen
            reward -= 10.0
        if abs(y) > 2.0:  # Too far sideways
            reward -= 5.0
        if vx < -0.5:  # Moving backward
            reward -= 2.0
        
        # 10. CURRICULUM SCALING
        reward *= (0.5 + 0.5 * self.difficulty)  # Scale rewards with difficulty
        
        return reward
    
    def _is_terminated(self):
        pos = self.data.qpos
        
        # More forgiving termination for easier training
        min_height = 0.4 / self.difficulty
        max_lateral = 3.0 / self.difficulty
        
        if pos[2] < min_height:
            return True
        if abs(pos[1]) > max_lateral:
            return True
        if pos[0] < -3.0:  # Moved too far backward
            return True
            
        return False
    
    def render(self, mode="human"):
        if self.viewer is None:
            try:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except:
                self.viewer = "disabled"
        
        if self.viewer != "disabled" and hasattr(self.viewer, 'sync'):
            self.viewer.sync()

def curriculum_training():
    """Train with curriculum learning - start easy, get harder"""
    print("🎓 CURRICULUM LEARNING APPROACH")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models/curriculum", exist_ok=True)
    os.makedirs("logs/curriculum", exist_ok=True)
    
    difficulties = [0.5, 0.7, 1.0, 1.3, 1.5]  # Easy to hard
    model = None
    
    for i, difficulty in enumerate(difficulties):
        print(f"\n🎯 STAGE {i+1}: Difficulty {difficulty}")
        print(f"Learning rate: {3e-4 / difficulty:.2e}")
        
        # Create environment with current difficulty
        env = DummyVecEnv([lambda d=difficulty: Monitor(ImprovedWalkingEnv(difficulty=d))])
        
        # Load previous model or create new one
        if model is not None:
            # Continue from previous stage
            model.set_env(env)
            print(f"✅ Continuing from previous stage")
        else:
            # Create new model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4 / difficulty,  # Easier = higher learning rate
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01 / difficulty,  # Easier = more exploration
                verbose=1,
                tensorboard_log=f"./logs/curriculum/",
                policy_kwargs=dict(
                    net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
                )
            )
            print(f"✅ Created new model for stage {i+1}")
        
        # Train for this stage
        timesteps = int(50000 * (1.0 + difficulty))  # Harder = more training
        print(f"Training for {timesteps} timesteps...")
        
        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"curriculum_stage_{i+1}_diff_{difficulty}"
        )
        
        # Save stage model
        model.save(f"models/curriculum/stage_{i+1}_difficulty_{difficulty}")
        
        # Test stage
        print(f"🧪 Testing stage {i+1}...")
        test_env = ImprovedWalkingEnv(difficulty=difficulty)
        obs, _ = test_env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        avg_reward = total_reward / steps
        final_distance = test_env.data.qpos[0]
        
        print(f"📊 Stage {i+1} Results:")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Distance traveled: {final_distance:.2f}m")
        print(f"   Steps survived: {steps}")
        
        if avg_reward < 2.0:
            print(f"⚠️  Stage {i+1} needs more training, extending...")
            # Extra training if not performing well
            model.learn(total_timesteps=25000, progress_bar=True, reset_num_timesteps=False)
    
    # Final model
    model.save("advanced_walking_model")
    print("✅ Curriculum training completed!")
    return model

def intensive_training():
    """Intensive training with the best model so far"""
    print("\n🔥 INTENSIVE TRAINING")
    print("=" * 30)
    
    # Load best existing model
    try:
        env = DummyVecEnv([lambda: Monitor(ImprovedWalkingEnv(difficulty=1.0))])
        model = PPO.load("walking_model", env=env)
        print("✅ Loaded existing walking model")
    except:
        env = DummyVecEnv([lambda: Monitor(ImprovedWalkingEnv(difficulty=1.0))])
        model = PPO(
            "MlpPolicy", env, learning_rate=1e-4, verbose=1,
            tensorboard_log="./logs/intensive/",
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]))
        )
        print("✅ Created new intensive model")
    
    # Intensive training
    print("🎯 Intensive training for 200,000 steps...")
    model.learn(
        total_timesteps=200000,
        progress_bar=True,
        tb_log_name="intensive_walking"
    )
    
    model.save("intensive_walking_model")
    print("✅ Intensive training completed!")
    return model

def main():
    print("🚀 ADVANCED WALKING TRAINING SUITE")
    print("=" * 50)
    
    choice = input("""
Choose training approach:
1. 🎓 Curriculum Learning (recommended - start easy, get harder)
2. 🔥 Intensive Training (more training with current model)
3. 🎯 Both (curriculum then intensive)

Enter choice (1/2/3): """).strip()
    
    if choice == "1":
        model = curriculum_training()
    elif choice == "2":
        model = intensive_training()
    elif choice == "3":
        model = curriculum_training()
        model = intensive_training()
    else:
        print("Invalid choice, using curriculum learning...")
        model = curriculum_training()
    
    print("\n🎬 To view tensorboard logs:")
    print("tensorboard --logdir=./logs/")
    print("\n🎮 To test the improved model:")
    print("mjpython -c \"")
    print("from advanced_walking_training import ImprovedWalkingEnv")
    print("from stable_baselines3 import PPO; import time")
    print("env = ImprovedWalkingEnv(); model = PPO.load('advanced_walking_model', env=env)")
    print("obs, _ = env.reset()")
    print("for i in range(2000):")
    print("    action, _ = model.predict(obs, deterministic=True)")
    print("    obs, r, t, tr, _ = env.step(action); env.render(); time.sleep(0.02)")
    print("    if t or tr: obs, _ = env.reset()\"")

if __name__ == "__main__":
    main() 