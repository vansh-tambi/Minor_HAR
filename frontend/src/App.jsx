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
    setStatus({ msg: 'Inferencing Kinetics...', type: 'ok' });
    try {
      const response = await axios.post(`${backendUrl}/predict`, { data: windowData });
      const result = response.data;
      
      setPrediction(result);
      setStatus({ msg: 'Connected. Broadcasting via TCP.', type: 'ok' });
    } catch (err) {
      console.error(err);
      setStatus({ msg: 'Warning: Cannot reach Deep Learning Backend', type: 'err' });
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
  const activity = prediction?.activity || 'AWAITING INPUT';
  const confidence = prediction?.confidence || 0;
  const pct = Math.round(confidence * 100) || 0;
  const raw_activity = prediction?.raw_activity;
  
  let confBackground = 'linear-gradient(90deg, var(--accent-magenta), var(--accent-red))';
  let confShadow = '0 0 16px var(--accent-red)';
  
  if (confidence >= 0.8) {
    confBackground = 'linear-gradient(90deg, var(--accent-green), #fff)';
    confShadow = '0 0 16px var(--accent-green)';
  } else if (confidence >= 0.6) {
    confBackground = 'linear-gradient(90deg, var(--accent-cyan), #fff)';
    confShadow = '0 0 16px var(--accent-cyan)';
  }
  
  const isUncertain = raw_activity === 'Uncertain' || activity === 'Uncertain';

  const probEntries = prediction?.all_probs 
    ? Object.entries(prediction.all_probs).sort((a, b) => b[1] - a[1]).slice(0, 5)
    : [];

  // Animation variants
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.1, duration: 0.5, ease: 'easeOut' }
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

      <motion.div className="header" initial={{opacity: 0, scale: 0.95}} animate={{opacity: 1, scale: 1}} transition={{duration: 0.6}}>
        <h1><Activity size={32} color="var(--accent-cyan)" /> HAR Engine</h1>
        <div className="subtitle">Real-Time Kinetic Intelligence Terminal</div>
      </motion.div>

      <div className="grid">
        {/* Status */}
        <motion.div className={`status-bar ${status.type}`} id="status" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 0.1}}>
            {status.type === 'ok' ? <CheckCircle2 size={16}/> : <AlertCircle size={16}/>}
            <span id="status-text">{status.msg}</span>
        </motion.div>

        {/* Activity Hero */}
        <motion.div className="card" id="activity-card" custom={0} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Target size={18} /> Deep Learning Prediction</h2>
          
          <AnimatePresence mode="wait">
            <motion.div 
              key={activity}
              className="activity-label"
              initial={{ opacity: 0, scale: 0.9, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 1.05, filter: 'blur(10px)' }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              {activity}
            </motion.div>
          </AnimatePresence>

          {prediction && (
            <motion.span className={`status-badge ${isUncertain ? 'uncertain' : 'normal'}`} initial={{opacity:0}} animate={{opacity:1}}>
              {isUncertain ? <AlertCircle size={14}/> : <CheckCircle2 size={14}/>}
              {isUncertain ? 'Low Confidence Alert' : 'Target Locked'}
            </motion.span>
          )}

          <div className="confidence-bar-wrap">
            <div className="bar-bg">
              <div className="bar-fill" style={{ width: `${pct}%`, background: confBackground, boxShadow: confShadow }}></div>
            </div>
            <div className="confidence-text">Match Fidelity: {prediction ? pct : '0'}%</div>
          </div>
        </motion.div>

        {/* Data Windowing / Progress */}
        <motion.div className="card" id="buffer-wrap" custom={1} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Radar size={18}/> Streaming Window</h2>
          <div className="confidence-text" style={{textAlign: 'left', marginBottom: '8px', color: 'var(--text-muted)'}}>{bufferSize} / {totalSize} Packets (<span style={{color: 'var(--accent-cyan)'}}>{isCapturing ? 'ONLINE' : 'OFFLINE'}</span>)</div>
          <div className="bar-bg slim">
            <div className="bar-fill" style={{ width: `${(bufferSize/totalSize)*100}%`, background: 'var(--accent-cyan)', boxShadow: 'none' }}></div>
          </div>
        </motion.div>

        {/* Live sensor values */}
        <motion.div className="card" id="sensors-card" custom={2} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Activity size={18}/> Accelerometer Vectors</h2>
          <div className="sensor-vals">
            <div className="axis x"><span>{currentData.ax.toFixed(2)}</span><label>X-Axis (ms²)</label></div>
            <div className="axis y"><span>{currentData.ay.toFixed(2)}</span><label>Y-Axis (ms²)</label></div>
            <div className="axis z"><span>{currentData.az.toFixed(2)}</span><label>Z-Axis (ms²)</label></div>
          </div>
          <h2 style={{marginTop: '24px'}}><Activity size={18}/> Gyroscope Rotations</h2>
          <div className="sensor-vals">
            <div className="axis x"><span>{currentData.gx.toFixed(2)}</span><label>Alpha (°/s)</label></div>
            <div className="axis y"><span>{currentData.gy.toFixed(2)}</span><label>Beta (°/s)</label></div>
            <div className="axis z"><span>{currentData.gz.toFixed(2)}</span><label>Gamma (°/s)</label></div>
          </div>
        </motion.div>

        {/* Top probabilities */}
        <motion.div className="card" id="prob-card" custom={3} initial="hidden" animate="visible" variants={cardVariants}>
          <h2><Target size={18}/> Neural Output Distribution</h2>
          <div id="prob-list">
            {probEntries.map(([name, prob], i) => (
                <div className="prob-row" key={name}>
                <div className="prob-name">{name}</div>
                <div className="prob-bar-bg">
                  <motion.div 
                    className="prob-bar" 
                    initial={{width: 0}}
                    animate={{width: `${prob*100}%`}}
                    transition={{duration: 0.5}}
                    style={{ background: i === 0 ? 'var(--accent-cyan)' : 'var(--text-muted)' }}
                  />
                </div>
                <div className="prob-pct">{Math.round(prob*100)}%</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Chart */}
        <motion.div className="card" id="chart-card" custom={4} initial="hidden" animate="visible" variants={cardVariants} style={{ paddingBottom: '0' }}>
          <h2><Activity size={18}/> Raw Data Waveforms</h2>
          <div style={{ height: '160px', position: 'relative' }}>
            <SensorChart data={currentData} />
          </div>
        </motion.div>

        {/* Controls */}
        <motion.div id="controls" custom={5} initial="hidden" animate="visible" variants={cardVariants}>
          <button className="btn-start" onClick={startBtnClick} disabled={isCapturing}>
            <Play size={20} fill="currentColor" /> Initiate Stream
          </button>
          <button className="btn-stop" onClick={stopBtnClick} disabled={!isCapturing}>
            <Square size={20} fill="currentColor" /> Terminate Link
          </button>
        </motion.div>

      </div>
    </>
  );
}

export default App;
