import React, { useState } from 'react';
import axios from 'axios';
import { useAccelerometer } from './hooks/useAccelerometer';
import SensorChart from './components/SensorChart';

function App() {
  const [backendUrl, setBackendUrl] = useState('http://localhost:5000');
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState({ msg: 'System Ready — Click Start to Begin', type: 'ok' });
  const [recentHistory, setRecentHistory] = useState([]);

  const onWindowReady = async (windowData) => {
    setStatus({ msg: 'Predicting…', type: 'ok' });
    try {
      const response = await axios.post(`${backendUrl}/predict`, { data: windowData });
      const result = response.data;
      
      setPrediction(result);
      setStatus({ msg: 'OK — predicting every 10 s', type: 'ok' });
      
      const ts = new Date().toLocaleTimeString();
      setRecentHistory(prev => {
        const next = [...prev, { activity: result.raw_activity || result.activity, confidence: result.confidence, timestamp: ts }];
        if (next.length > 3) next.shift();
        return next;
      });
    } catch (err) {
      console.error(err);
      setStatus({ msg: 'Cannot reach backend — is python backend/app.py running?', type: 'err' });
    }
  };

  const {
    isCapturing,
    currentData,
    bufferSize,
    totalSize,
    isSimulation,
    startCapture,
    stopCapture,
  } = useAccelerometer(backendUrl, onWindowReady);

  const startBtnClick = () => {
    startCapture();
    setStatus({ msg: 'Capturing…', type: 'ok' });
  };

  const stopBtnClick = () => {
    stopCapture();
    setStatus({ msg: 'Stopped', type: '' });
  };

  // derived values
  const activity = prediction?.activity || '—';
  const confidence = prediction?.confidence || 0;
  const pct = Math.round(confidence * 100) || 0;
  const raw_activity = prediction?.raw_activity;
  
  let confBackground = 'var(--secondary)';
  let confBoxShadow = '4px 0 12px rgba(207, 92, 255, 0.4)';
  if (confidence >= 0.8) {
    confBackground = 'var(--tertiary)';
    confBoxShadow = '4px 0 12px rgba(0, 247, 166, 0.4)';
  } else if (confidence >= 0.6) {
    confBackground = 'var(--primary)';
    confBoxShadow = '4px 0 12px rgba(0, 240, 255, 0.4)';
  }
  
  const isUncertain = raw_activity === 'Uncertain' || activity === 'Uncertain';

  const probEntries = prediction?.all_probs 
    ? Object.entries(prediction.all_probs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
    : [];

  return (
    <>
      <div className="header">
        <h1>🏃 Human Activity Recognition</h1>
        <div className="subtitle">Real-Time Kinetic Intelligence Dashboard</div>
      </div>

      <div style={{ position: 'absolute', top: '10px', right: '10px', display: 'flex', gap: '5px', zIndex: 100 }}>
        <input 
            type="text" 
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            style={{ 
                background: 'rgba(28, 27, 30, 0.8)', 
                border: '1px solid rgba(59, 73, 75, 0.5)', 
                color: 'var(--on-surface)',
                fontFamily: 'var(--font-display)',
                fontSize: '0.7rem',
                padding: '4px 8px',
                borderRadius: '4px',
                width: '160px'
            }}
            placeholder="Backend URL"
            title="Backend API base URL"
        />
      </div>

      <div className="grid">
        {/* Status */}
        <div className={`status-bar ${status.type}`} id="status">
          <span className="status-dot"></span>
          <span id="status-text">{status.msg}</span>
        </div>

        {/* Activity Hero */}
        <div className="card" id="activity-card">
          <span className="coord-label">ACT_01 // LIVE</span>
          <h2>Detected Activity</h2>
          <div className="activity-label">{activity}</div>
          
          {prediction && (
            <span className={`status-badge ${isUncertain ? 'uncertain' : 'normal'}`}>
              {isUncertain ? 'Uncertain' : 'Normal'}
            </span>
          )}
          
          <div className="confidence-bar-wrap">
            <div className="bar-bg">
              <div 
                className="bar-fill" 
                style={{
                  width: `${pct}%`,
                  background: confBackground,
                  boxShadow: confBoxShadow
                }}
              ></div>
            </div>
            <div className="confidence-text">Confidence: {prediction ? pct : '—'}%</div>
          </div>
        </div>

        {/* Controls */}
        <div id="controls">
          <button className="btn-start" onClick={startBtnClick} disabled={isCapturing}>
            ▶ Start Recording
          </button>
          <button className="btn-stop" onClick={stopBtnClick} disabled={!isCapturing}>
            ■ Stop Recording
          </button>
        </div>

        {/* Buffer Progress */}
        <div className="card" id="buffer-wrap">
          <span className="coord-label">BUF_02 // STREAM</span>
          <h2>Collecting Window</h2>
          <div className="buffer-label">{bufferSize} / {totalSize} samples</div>
          <div className="bar-bg slim">
            <div className="buffer-fill" style={{ width: `${(bufferSize/totalSize)*100}%` }}></div>
          </div>
        </div>

        {/* Live sensor values */}
        <div className="card">
          <span className="coord-label">SEN_03 // XYZ</span>
          <h2>Accelerometer (m/s²)</h2>
          <div className="sensor-vals">
            <div className="axis x"><span>{currentData.x.toFixed(2)}</span><label>X-Axis</label></div>
            <div className="axis y"><span>{currentData.y.toFixed(2)}</span><label>Y-Axis</label></div>
            <div className="axis z"><span>{currentData.z.toFixed(2)}</span><label>Z-Axis</label></div>
          </div>
        </div>

        {/* Chart */}
        <div className="card" style={{ paddingBottom: '0' }}>
          <span className="coord-label">VIS_04 // WAVEFORM</span>
          <h2>Sensor Stream</h2>
          <div style={{ height: '140px', position: 'relative' }}>
            <SensorChart data={currentData} />
          </div>
        </div>

        {/* Top probabilities */}
        <div className="card" id="prob-card">
          <span className="coord-label">PRB_05 // DIST</span>
          <h2>Probability Distribution</h2>
          <div id="prob-list">
            {probEntries.map(([name, prob]) => (
                <div className="prob-row" key={name}>
                <div className="prob-name">{name}</div>
                <div className="prob-bar-bg">
                  <div className="prob-bar" style={{ width: `${Math.round(prob*100)}%` }}></div>
                </div>
                <div className="prob-pct">{Math.round(prob*100)}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent predictions */}
        <div className="card" id="history-card">
          <span className="coord-label">HST_06 // LOG</span>
          <h2>Recent Predictions (Smoothing)</h2>
          <div className="history-list">
            {recentHistory.length === 0 ? (
                <div className="history-empty">No predictions yet</div>
            ) : (
                [...recentHistory].reverse().map((item, i) => (
                    <div className={`history-item ${i === 0 ? 'newest' : ''}`} key={i}>
                      <span className="h-label">{item.activity}</span>
                      <span className="h-conf">{Math.round(item.confidence * 100)}% · {item.timestamp}</span>
                    </div>
                ))
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
