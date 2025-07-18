import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

class HumanoidDemoEnv(gym.Env):
    def __init__(self, xml_path, demo_states, demo_actions, cfg):
        super().__init__()
        self.model   = load_model_from_path(xml_path)
        self.sim     = MjSim(self.model)
        self.viewer  = MjViewer(self.sim)
        self.n_qpos  = self.sim.model.nq
        self.n_ctrl  = self.sim.model.nu

        # demo data & config
        self.demo_s = demo_states
        self.demo_a = demo_actions
        self.cfg    = cfg
        self.demo_idx = 0
        self.use_imitation = True

        high = np.inf * np.ones(self.n_qpos + self.sim.model.nv)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, shape=(self.n_ctrl,), dtype=np.float32)

    def reset(self):
        s0 = self.demo_s[self.demo_idx]
        self.sim.data.qpos[:] = s0
        self.sim.data.qvel[:] = 0
        self.sim.forward()
        return self._get_obs()

    def step(self, action):
        if self.use_imitation:
            action = self.demo_a[self.demo_idx]

        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = False

        self.demo_idx = (self.demo_idx + 1) % len(self.demo_s)
        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    def _compute_reward(self, obs, action):
        vel = self.sim.data.qvel[0]
        roll, pitch = self.sim.data.qpos[3], self.sim.data.qpos[4]
        r_vel     = self.cfg["w_vel"] * vel
        r_upright = -self.cfg["w_upright"] * (abs(roll) + abs(pitch))
        r_energy  = -self.cfg["w_energy"] * np.sum(np.square(action))
        return r_vel + r_upright + r_energy

    def render(self, mode="human"):
        if self.viewer.is_alive and self.viewer.window:
            key = self.viewer.get_key()
            if key in (ord('T'), ord('t')):
                self.use_imitation = not self.use_imitation
                print(f"Imitation mode: {self.use_imitation}")
        self.viewer.render()
