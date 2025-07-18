import mujoco
import numpy as np

class MujocoSimulator:
    def __init__(self, xml_path="humanoid.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

    def set_joint_angles(self, joint_angles):
        """Set joint angles from motion data."""
        for i in range(min(len(joint_angles), self.model.nq)):
            self.data.qpos[i] = joint_angles[i]

    def step(self):
        """Step the simulation."""
        mujoco.mj_step(self.model, self.data)

    def render(self, width=800, height=600):
        """Render the simulation."""
        self.renderer.update_scene(self.data)
        return self.renderer.render(width, height)

if __name__ == "__main__":
    sim = MujocoSimulator()
    dummy_angles = np.random.rand(sim.model.nq)  # Random joint angles
    sim.set_joint_angles(dummy_angles)
    sim.step()
    img = sim.render()
    print(f"Rendered image shape: {img.shape}")