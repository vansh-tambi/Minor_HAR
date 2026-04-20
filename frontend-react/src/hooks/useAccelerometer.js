import { useState, useEffect, useRef } from 'react';

const FRAME_SIZE = 200;
const FREQ = 20; // Hz
const INTERVAL = 1000 / FREQ;

export const useAccelerometer = (backendUrl, onWindowReady) => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentData, setCurrentData] = useState({ x: 0, y: 0, z: 0 });
  const [bufferSize, setBufferSize] = useState(0);
  const [isSimulation, setIsSimulation] = useState(false);
  const [error, setError] = useState(null);

  const bufferRef = useRef([]);
  const intervalIdRef = useRef(null);
  const sensorRef = useRef({ x: 0, y: 0, z: 0 });

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
    if (acc) {
      sensorRef.current = {
        x: acc.x || 0,
        y: acc.y || 0,
        z: acc.z || 0,
      };
    }
  };

  const tick = () => {
    let { x, y, z } = sensorRef.current;

    if (isSimulation || !window.DeviceMotionEvent) {
      // Simple simulation for desktop
      const t = Date.now() / 1000;
      x = Math.sin(t * 1.5) * 0.5 + (Math.random() - 0.5) * 0.1;
      y = 9.8 + Math.cos(t * 0.8) * 0.3 + (Math.random() - 0.5) * 0.1;
      z = Math.sin(t * 1.1) * 0.4 + (Math.random() - 0.5) * 0.1;
      sensorRef.current = { x, y, z };
    }

    setCurrentData({ x, y, z });
    bufferRef.current.push([x, y, z]);
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
