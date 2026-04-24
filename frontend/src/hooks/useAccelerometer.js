import { useState, useEffect, useRef } from 'react';

const FRAME_SIZE = 60;
const FREQ = 20; // Hz
const INTERVAL = 1000 / FREQ;

export const useAccelerometer = (backendUrl, onWindowReady) => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentData, setCurrentData] = useState({ ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
  const [bufferSize, setBufferSize] = useState(0);
  const [isSimulation, setIsSimulation] = useState(false);
  const [error, setError] = useState(null);

  const bufferRef = useRef([]);
  const intervalIdRef = useRef(null);
  const sensorRef = useRef({ ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });

  const startCapture = async () => {
    if (isCapturing) return;

    try {
      // Permission Handling (iOS 13+)
      if (
        typeof DeviceMotionEvent !== 'undefined' &&
        typeof DeviceMotionEvent.requestPermission === 'function'
      ) {
        const permission = await DeviceMotionEvent.requestPermission();
        if (permission !== 'granted') {
          throw new Error('Permission to access motion sensors was denied.');
        }
      }

      const hasSensor = window.DeviceMotionEvent;
      if (!hasSensor) {
        setIsSimulation(true);
      } else {
        window.addEventListener('devicemotion', handleMotion);
      }

      setIsCapturing(true);
      setError(null);
      bufferRef.current = [];
      setBufferSize(0);

      // Start the collection loop
      intervalIdRef.current = setInterval(tick, INTERVAL);
    } catch (err) {
      setError(err.message);
      setIsSimulation(true); // Fallback to simulation
      setIsCapturing(true);
      intervalIdRef.current = setInterval(tick, INTERVAL);
    }
  };

  const stopCapture = () => {
    setIsCapturing(false);
    if (intervalIdRef.current) clearInterval(intervalIdRef.current);
    window.removeEventListener('devicemotion', handleMotion);
    bufferRef.current = [];
    setBufferSize(0);
  };

  const handleMotion = (e) => {
    const acc = e.accelerationIncludingGravity;
    // Gyroscope data (rotationRate gives angular velocity in deg/s)
    const rot = e.rotationRate;
    
    // Neural Network is trained on gyroscope data in rad/s.
    const DEG_TO_RAD = Math.PI / 180;

    sensorRef.current = {
      ax: acc?.x || 0,
      ay: acc?.y || 0,
      az: acc?.z || 0,
      // API documentation: 
      // beta = X-axis, gamma = Y-axis, alpha = Z-axis
      gx: (rot?.beta || 0) * DEG_TO_RAD,
      gy: (rot?.gamma || 0) * DEG_TO_RAD,
      gz: (rot?.alpha || 0) * DEG_TO_RAD,
    };
  };

  const tick = () => {
    let { ax, ay, az, gx, gy, gz } = sensorRef.current;

    if (isSimulation || !window.DeviceMotionEvent) {
      // Desktop simulation: generate realistic-looking sensor data
      const t = Date.now() / 1000;

      // Simulated accelerometer (walking-like pattern)
      ax = Math.sin(t * 1.5) * 0.5 + (Math.random() - 0.5) * 0.1;
      ay = 9.8 + Math.cos(t * 0.8) * 0.3 + (Math.random() - 0.5) * 0.1;
      az = Math.sin(t * 1.1) * 0.4 + (Math.random() - 0.5) * 0.1;

      // Simulated gyroscope
      gx = Math.sin(t * 2.0) * 0.3 + (Math.random() - 0.5) * 0.05;
      gy = Math.cos(t * 1.3) * 0.2 + (Math.random() - 0.5) * 0.05;
      gz = Math.sin(t * 0.7) * 0.15 + (Math.random() - 0.5) * 0.05;

      sensorRef.current = { ax, ay, az, gx, gy, gz };
    }

    setCurrentData({ ax, ay, az, gx, gy, gz });

    // Send 6 values per sample: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    bufferRef.current.push([ax, ay, az, gx, gy, gz]);
    setBufferSize(bufferRef.current.length);

    if (bufferRef.current.length >= FRAME_SIZE) {
      const windowData = [...bufferRef.current];
      bufferRef.current = [];
      setBufferSize(0);
      onWindowReady(windowData);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalIdRef.current) clearInterval(intervalIdRef.current);
      window.removeEventListener('devicemotion', handleMotion);
    };
  }, []);

  return {
    isCapturing,
    currentData,
    bufferSize,
    totalSize: FRAME_SIZE,
    isSimulation,
    error,
    startCapture,
    stopCapture,
  };
};
