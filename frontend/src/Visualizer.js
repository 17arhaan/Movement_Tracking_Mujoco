import React, { useState } from 'react';

function Visualizer() {
  const [image, setImage] = useState(null);

  const simulateMotion = async () => {
    const joint_angles = Array(17).fill(0); // Replace with actual motion data
    const response = await fetch('http://localhost:5000/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ joint_angles }),
    });
    const data = await response.json();
    setImage(`data:image/png;base64,${btoa(String.fromCharCode(...data.image.flat()))}`);
  };

  return (
    <div>
      <button onClick={simulateMotion}>Simulate Motion</button>
      {image && <img src={image} alt="Simulation" />}
    </div>
  );
}

export default Visualizer;