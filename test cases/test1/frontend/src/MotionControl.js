import React, { useState } from 'react';

function MotionControl() {
  const [status, setStatus] = useState('Ready to load motion data...');
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setStatus('Loading motion data...');
    try {
      const response = await fetch('http://localhost:5000/load_data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setStatus(`✅ Loaded sequences with shape: ${data.sequences_shape}`);
    } catch (error) {
      setStatus(`❌ Error loading data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    setLoading(true);
    setStatus('Training LSTM model...');
    try {
      // Generate some dummy training data
      const sequences = Array(100).fill().map(() => 
        Array(50).fill().map(() => Array(17).fill().map(() => Math.random()))
      );
      const labels = Array(100).fill().map(() => 
        Array(6).fill(0).map((_, i) => i === Math.floor(Math.random() * 6) ? 1 : 0)
      );
      
      const response = await fetch('http://localhost:5000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequences, labels }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setStatus(`✅ ${result.message}`);
    } catch (error) {
      setStatus(`❌ Error training model: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="control-panel">
      <h2>Motion Control</h2>
      <button 
        className="button" 
        onClick={loadData} 
        disabled={loading}
      >
        {loading ? 'Loading...' : 'Load Motion Data'}
      </button>
      <button 
        className="button" 
        onClick={trainModel} 
        disabled={loading}
      >
        {loading ? 'Training...' : 'Train LSTM'}
      </button>
      <div className={`status ${loading ? 'loading' : ''}`}>
        {status}
      </div>
    </div>
  );
}

export default MotionControl;