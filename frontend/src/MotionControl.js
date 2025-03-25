import React, { useState } from 'react';

function MotionControl() {
  const [status, setStatus] = useState('');

  const loadData = async () => {
    const response = await fetch('http://localhost:5000/load_data');
    const data = await response.json();
    setStatus(`Loaded sequences: ${data.sequences_shape}`);
  };

  const trainModel = async () => {
    const sequences = []; // Replace with actual data
    const labels = [];    // Replace with actual labels
    const response = await fetch('http://localhost:5000/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequences, labels }),
    });
    const result = await response.json();
    setStatus(result.message);
  };

  return (
    <div>
      <button onClick={loadData}>Load Motion Data</button>
      <button onClick={trainModel}>Train LSTM</button>
      <p>{status}</p>
    </div>
  );
}

export default MotionControl;