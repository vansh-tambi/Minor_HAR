import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const MAX_SAMPLES = 60;

const SensorChart = ({ data }) => {
  const chartRef = useRef(null);
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { display: false },
      y: {
        grid: { color: 'rgba(59, 73, 75, 0.1)' },
        ticks: { color: '#849495', font: { size: 10 } }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#849495',
          boxWidth: 8,
          usePointStyle: true,
          font: { family: 'Space Grotesk' }
        }
      }
    }
  };

  const chartData = useRef({
    labels: Array(MAX_SAMPLES).fill(''),
    datasets: [
      {
        label: 'X-Axis',
        data: Array(MAX_SAMPLES).fill(0),
        borderColor: '#ff6b8a',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
      },
      {
        label: 'Y-Axis',
        data: Array(MAX_SAMPLES).fill(0),
        borderColor: '#00f7a6',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
      },
      {
        label: 'Z-Axis',
        data: Array(MAX_SAMPLES).fill(0),
        borderColor: '#00f0ff',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
      },
    ],
  });

  useEffect(() => {
    const { datasets } = chartData.current;
    datasets[0].data.push(data.x);
    datasets[1].data.push(data.y);
    datasets[2].data.push(data.z);

    if (datasets[0].data.length > MAX_SAMPLES) {
      datasets[0].data.shift();
      datasets[1].data.shift();
      datasets[2].data.shift();
    }

    if (chartRef.current) {
      chartRef.current.update('none');
    }
  }, [data]);

  return <Line ref={chartRef} options={options} data={chartData.current} />;
};

export default SensorChart;
