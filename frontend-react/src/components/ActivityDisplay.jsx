import React from 'react';
import { motion } from 'framer-motion';

const ActivityDisplay = ({ prediction }) => {
  if (!prediction) {
    return (
      <div className="glass-card p-6 text-center h-full flex flex-col justify-center items-center">
        <h2 className="text-xs uppercase tracking-widest text-on-surface-muted mb-4">Current Prediction</h2>
        <p className="text-3xl font-bold opacity-30">OFFLINE</p>
      </div>
    );
  }

  const { activity, confidence, raw_activity } = prediction;
  const pct = Math.round(confidence * 100);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 h-full flex flex-col justify-between"
    >
      <div>
        <div className="flex justify-between items-start mb-2">
          <h2 className="text-xs uppercase tracking-widest text-on-surface-muted">Activity Detected</h2>
          <span className={`px-2 py-1 rounded text-[10px] font-bold tracking-tighter uppercase ${
            activity === 'Uncertain' ? 'bg-secondary/10 text-secondary border border-secondary/20' : 'bg-tertiary/10 text-tertiary border border-tertiary/20'
          }`}>
            {raw_activity === 'Uncertain' ? 'Low Signal' : 'Active'}
          </span>
        </div>
        <motion.h1 
          key={activity}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-4xl font-bold text-gradient mb-4"
        >
          {activity}
        </motion.h1>
      </div>

      <div>
        <div className="flex justify-between text-xs mb-2">
          <span className="text-on-surface-muted">Confidence</span>
          <span className="font-mono">{pct}%</span>
        </div>
        <div className="h-2 bg-surface-highest rounded-full overflow-hidden">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="h-full bg-primary"
            style={{ 
              boxShadow: '0 0 12px var(--primary)',
              background: pct > 80 ? 'var(--tertiary)' : 'var(--primary)'
            }}
          />
        </div>
      </div>
    </motion.div>
  );
};

export default ActivityDisplay;
