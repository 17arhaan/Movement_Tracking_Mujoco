from flask import Flask, jsonify, request
from motion_tracker import MotionTracker
from lstm_model import LSTMModel
from mujoco_sim import MujocoSimulator
import numpy as np

app = Flask(__name__)
tracker = MotionTracker()
model = LSTMModel(input_shape=(50, 17))
sim = MujocoSimulator()

@app.route('/load_data', methods=['GET'])
def load_data():
    data = tracker.load_data()
    sequences = tracker.preprocess()
    return jsonify({"sequences_shape": sequences.shape})

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    X = np.array(data['sequences'])
    y = np.array(data['labels'])
    history = model.train(X, y)
    model.save("lstm_model.h5")
    return jsonify({"message": "Model trained", "history": history.history})

@app.route('/simulate', methods=['POST'])
def simulate():
    joint_angles = request.json['joint_angles']
    sim.set_joint_angles(joint_angles)
    sim.step()
    img = sim.render()
    return jsonify({"image": img.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)