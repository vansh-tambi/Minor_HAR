import React, { useState } from 'react';
import axios from 'axios';
import { useAccelerometer } from './hooks/useAccelerometer';
import SensorChart from './components/SensorChart';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Activity, Play, Square, AlertCircle, CheckCircle2, Radar, Target } from 'lucide-react';

function App() {
  const [backendUrl, setBackendUrl] = useState(import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000');
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState({ msg: 'System Ready', type: 'ok' });

  const onWindowReady = async (windowData) => {
    setStatus({ msg: 'Processing data...', type: 'ok' });
    try {
      const response = await axios.post(`${backendUrl}/predict`, { data: windowData });
      const result = response.data;
      
      setPrediction(result);
      setStatus({ msg: 'Connected to Prediction Service.', type: 'ok' });
    } catch (err) {
      console.error(err);
      setStatus({ msg: 'Error: Cannot reach backend service', type: 'err' });
    }
  };

  const {
    isCapturing,
    currentData,
    bufferSize,
    totalSize,
    startCapture,
    stopCapture,
  } = useAccelerometer(backendUrl, onWindowReady);

  const startBtnClick = () => {
    startCapture();
    setStatus({ msg: 'Data stream initiated...', type: 'ok' });
  };

  const stopBtnClick = () => {
    stopCapture();
    setStatus({ msg: 'Stream offline', type: '' });
  };

  // derived values
  const activity = prediction?.activity || 'AWAITING DATA';
  const confidence = prediction?.confidence || 0;
  const pct = Math.round(confidence * 100) || 0;
  const raw_activity = prediction?.raw_activity;
  
  let confBackground = 'var(--secondary-color)';
  
  if (confidence >= 0.8) {
    confBackground = 'var(--success-color)';
  } else if (confidence >= 0.6) {
    confBackground = 'var(--primary-color)';
  } else {
    confBackground = 'var(--warning-color)';
  }
  
  const isUncertain = raw_activity === 'Uncertain' || activity === 'Uncertain';

  const probEntries = prediction?.all_probs 
    ? Object.entries(prediction.all_probs).sort((a, b) => b[1] - a[1]).slice(0, 5)
    : [];

  // Animation variants
  const cardVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.05, duration: 0.3, ease: 'easeOut' }
    })
  };

  return (
    <>
      <div style={{ position: 'fixed', top: '16px', right: '16px', display: 'flex', alignItems: 'center', gap: '8px', zIndex: 100 }}>
        <Network size={16} color="var(--text-muted)" />
        <input 
            type="text" 
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            style={{ 
                background: 'var(--input-bg)', 
                border: '1px solid var(--card-border)', 
                color: 'var(--text-main)',
                fontFamily: 'var(--font-body)',
                fontSize: '0.85rem',
                padding: '8px 12px',
                borderRadius: '6px',
                width: '180px',
                outline: 'none'
            }}
            placeholder="Backend URL"
        />
      </div>

      <motion.div className="header" initial={{opacity: 0, y: -10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.4}}>
        <h1><Activity size={28} color="var(--primary-color)" /> Activity Recognition</h1>
        <div className="subtitle">Real-time human activity monitoring dashboard</div>
      </motion.div>

      <div className="grid">
        {/* Status */}
        <motion.div className={`status-bar ${status.type}`} id="status" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 0.1}}>
            {status.type === 'ok' ? <CheckCircle2 size={16}/> : <AlertCircle size={16}/>}
            <span id="status-text">{status.msg}</span>
        </motion.div>

        {/* Activity Hero */}
        <motion.div className="card" id="activity-card" custom={0} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Target size={18} /> Prediction</h2>
          
          <AnimatePresence mode="wait">
            <motion.div 
              key={activity}
              className="activity-label"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {activity}
            </motion.div>
          </AnimatePresence>

          {prediction && (
            <motion.span className={`status-badge ${isUncertain ? 'uncertain' : 'normal'}`} initial={{opacity:0}} animate={{opacity:1}}>
              {isUncertain ? <AlertCircle size={14}/> : <CheckCircle2 size={14}/>}
              {isUncertain ? 'Low Confidence' : 'Confident Prediction'}
            </motion.span>
          )}

          <div className="confidence-bar-wrap">
            <div className="bar-bg">
              <div className="bar-fill" style={{ width: `${pct}%`, background: confBackground }}></div>
            </div>
            <div className="confidence-text">Confidence Score: {prediction ? pct : '0'}%</div>
          </div>
        </motion.div>

        {/* Data Windowing / Progress */}
        <motion.div className="card" id="buffer-wrap" custom={1} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Radar size={18}/> Buffer Status</h2>
          <div className="confidence-text" style={{textAlign: 'left', marginBottom: '8px', color: 'var(--text-muted)'}}>{bufferSize} / {totalSize} Samples (<span style={{color: 'var(--primary-color)'}}>{isCapturing ? 'Online' : 'Offline'}</span>)</div>
          <div className="bar-bg slim">
            <div className="bar-fill" style={{ width: `${(bufferSize/totalSize)*100}%`, background: 'var(--primary-color)' }}></div>
          </div>
        </motion.div>

        {/* Live sensor values */}
        <motion.div className="card" id="sensors-card" custom={2} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Activity size={18}/> Accelerometer</h2>
          <div className="sensor-vals">
            <div className="axis x"><span>{currentData.ax.toFixed(2)}</span><label>X-Axis (m/s²)</label></div>
            <div className="axis y"><span>{currentData.ay.toFixed(2)}</span><label>Y-Axis (m/s²)</label></div>
            <div className="axis z"><span>{currentData.az.toFixed(2)}</span><label>Z-Axis (m/s²)</label></div>
          </div>
          <h2 style={{marginTop: '24px'}}><Activity size={18}/> Gyroscope</h2>
          <div className="sensor-vals">
            <div className="axis x"><span>{currentData.gx.toFixed(2)}</span><label>X-Axis (rad/s)</label></div>
            <div className="axis y"><span>{currentData.gy.toFixed(2)}</span><label>Y-Axis (rad/s)</label></div>
            <div className="axis z"><span>{currentData.gz.toFixed(2)}</span><label>Z-Axis (rad/s)</label></div>
          </div>
        </motion.div>

        {/* Top probabilities */}
        <motion.div className="card" id="prob-card" custom={3} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Target size={18}/> Class Probabilities</h2>
          <div id="prob-list">
            {probEntries.map(([name, prob], i) => (
                <div className="prob-row" key={name}>
                <div className="prob-name">{name}</div>
                <div className="prob-bar-bg">
                  <motion.div 
                    className="prob-bar" 
                    initial={{width: 0}}
                    animate={{width: `${prob*100}%`}}
                    transition={{duration: 0.3}}
                    style={{ background: i === 0 ? 'var(--primary-color)' : 'var(--secondary-color)' }}
                  />
                </div>
                <div className="prob-pct">{Math.round(prob*100)}%</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Chart */}
        <motion.div className="card" id="chart-card" custom={4} initial="hidden" animate="visible" variants={cardVariants} style={{ paddingBottom: '0' }}>
          <h2><Activity size={18}/> Live Sensor Data</h2>
          <div style={{ height: '160px', position: 'relative' }}>
            <SensorChart data={currentData} />
          </div>
        </motion.div>

        {/* Controls */}
        <motion.div id="controls" custom={5} initial="hidden" animate="visible" variants={cardVariants}>
          <button className="btn-start" onClick={startBtnClick} disabled={isCapturing}>
            <Play size={20} fill="currentColor" /> Start Recording
          </button>
          <button className="btn-stop" onClick={stopBtnClick} disabled={!isCapturing}>
            <Square size={20} fill="currentColor" /> Stop
          </button>
        </motion.div>

      </div>
    </>
  );
}

export default App;
