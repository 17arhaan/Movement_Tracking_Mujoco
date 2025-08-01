import gymnasium as gym
import numpy as np
import mujoco
import platform
import sys

class HumanoidDemoEnv(gym.Env):
    def __init__(self, xml_path, demo_states, demo_actions, cfg):
        super().__init__()
        self.model   = mujoco.MjModel.from_xml_path(xml_path)
        self.data    = mujoco.MjData(self.model)
        self.viewer  = None  # Will be initialized when render is called
        self.n_qpos  = self.model.nq
        self.n_ctrl  = self.model.nu

        # demo data & config
        self.demo_s = demo_states
        self.demo_a = demo_actions
        self.cfg    = cfg
        self.demo_idx = 0
        self.use_imitation = True

        # Create observation space with proper dimensions
        # Observation includes qpos (nq) + qvel (nv) - these can be different!
        obs_dim = self.model.nq + self.model.nv
        high = np.inf * np.ones(obs_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, shape=(self.n_ctrl,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        s0 = self.demo_s[self.demo_idx]
        
        # Handle dimension mismatch: pad or truncate as needed
        if len(s0) < self.model.nq:
            # Demo data has fewer dimensions - pad with zeros
            padded_qpos = np.zeros(self.model.nq)
            padded_qpos[:len(s0)] = s0
            self.data.qpos[:] = padded_qpos
            print(f"⚠️  Demo data has {len(s0)} dims, model expects {self.model.nq}. Padding with zeros.")
        elif len(s0) > self.model.nq:
            # Demo data has more dimensions - truncate
            self.data.qpos[:] = s0[:self.model.nq]
            print(f"⚠️  Demo data has {len(s0)} dims, model expects {self.model.nq}. Truncating.")
        else:
            # Perfect match
            self.data.qpos[:] = s0
            
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        if self.use_imitation:
            action = self.demo_a[self.demo_idx]

        # Handle action dimension mismatch: pad or truncate as needed
        if len(action) < self.model.nu:
            # Demo action has fewer dimensions - pad with zeros
            padded_action = np.zeros(self.model.nu)
            padded_action[:len(action)] = action
            self.data.ctrl[:] = padded_action
        elif len(action) > self.model.nu:
            # Demo action has more dimensions - truncate
            self.data.ctrl[:] = action[:self.model.nu]
        else:
            # Perfect match
            self.data.ctrl[:] = action
            
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs, action if len(action) <= self.model.nu else action[:self.model.nu])
        terminated = False
        truncated = False

        self.demo_idx = (self.demo_idx + 1) % len(self.demo_s)
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Return concatenated position and velocity
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self, obs, action):
        vel = self.data.qvel[0]
        roll, pitch = self.data.qpos[3], self.data.qpos[4]
        r_vel     = float(self.cfg["w_vel"]) * vel
        r_upright = -float(self.cfg["w_upright"]) * (abs(roll) + abs(pitch))
        r_energy  = -float(self.cfg["w_energy"]) * np.sum(np.square(action))
        return r_vel + r_upright + r_energy

    def render(self, mode="human"):
        if self.viewer is None:
            try:
                # Import viewer module
                import mujoco.viewer as viewer
                # Try to create the viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception as e:
                if platform.system() == "Darwin":  # macOS
                    print("WARNING: MuJoCo viewer on macOS requires running with 'mjpython' instead of 'python'")
                    print("To use the viewer, run your script with: mjpython your_script.py")
                    print("For now, rendering will be skipped.")
                    self.viewer = "disabled"  # Mark as disabled
                    return None
                else:
                    print(f"Error creating viewer: {e}")
                    self.viewer = "disabled"
                    return None
        
        # Only sync if viewer is properly initialized
        if self.viewer != "disabled" and hasattr(self.viewer, 'sync'):
            self.viewer.sync()
        
        return None
