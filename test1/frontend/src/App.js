import React from 'react';
import MotionControl from './MotionControl';
import Visualizer from './Visualizer';
import './App.css';

function App() {
  return (
    <div className="App">
      <h1>Human Motion Tracking</h1>
      <MotionControl />
      <Visualizer />
    </div>
  );
}

export default App;