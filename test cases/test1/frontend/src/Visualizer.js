import React, { useState } from 'react';

function Visualizer() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const simulateMotion = async () => {
    setLoading(true);
    setError('');
    setImage(null);
    
    try {
      // Generate random joint angles for simulation
      const joint_angles = Array(17).fill().map(() => Math.random() * 0.5 - 0.25);
      
      const response = await fetch('http://localhost:5000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ joint_angles }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Convert the image array to base64
      const imageArray = new Uint8Array(data.image.flat());
      const blob = new Blob([imageArray], { type: 'image/png' });
      const imageUrl = URL.createObjectURL(blob);
      setImage(imageUrl);
      
    } catch (error) {
      setError(`Error simulating motion: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="visualization-panel">
      <h2>MuJoCo Simulation</h2>
      <button 
        className="button" 
        onClick={simulateMotion}
        disabled={loading}
      >
        {loading ? 'Simulating...' : 'Simulate Motion'}
      </button>
      
      {error && (
        <div className="status error">
          ❌ {error}
        </div>
      )}
      
      {loading && (
        <div className="status loading">
          🔄 Generating simulation...
        </div>
      )}
      
      {image && (
        <div style={{ marginTop: '20px' }}>
          <img 
            src={image} 
            alt="MuJoCo Simulation" 
            className="simulation-image"
            onLoad={() => {
              // Clean up the blob URL when image is loaded
              if (image.startsWith('blob:')) {
                URL.revokeObjectURL(image);
              }
            }}
          />
        </div>
      )}
      
      {!image && !loading && !error && (
        <div className="status">
          Click "Simulate Motion" to see the humanoid simulation
        </div>
      )}
    </div>
  );
}

export default Visualizer;