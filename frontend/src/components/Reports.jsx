import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { FileText, Share2, Loader2, Sparkles, UserPlus, BarChart3 } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const ACTIVITY_COLORS = {
  "Walking": "rgba(54, 162, 235, 0.8)",
  "Jogging": "rgba(255, 99, 132, 0.8)",
  "Stairs": "rgba(255, 206, 86, 0.8)",
  "Still": "rgba(75, 192, 192, 0.8)",
  "Eating": "rgba(153, 102, 255, 0.8)",
  "Hand Activity": "rgba(255, 159, 64, 0.8)",
  "Sports": "rgba(201, 203, 207, 0.8)",
};

const Reports = ({ backendUrl, authToken }) => {
  const [reports, setReports] = useState({ own: [], shared: [] });
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [shareEmail, setShareEmail] = useState('');
  const [shareTarget, setShareTarget] = useState(null);
  
  const [activeTab, setActiveTab] = useState('my_reports');
  const [viewedSharedIds, setViewedSharedIds] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('viewedSharedIds') || '[]');
    } catch {
      return [];
    }
  });

  const unreadCount = reports.shared.filter(r => !viewedSharedIds.includes(r._id)).length;

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'shared_reports' && reports.shared.length > 0) {
      const allSharedIds = reports.shared.map(r => r._id);
      const newViewed = [...new Set([...viewedSharedIds, ...allSharedIds])];
      setViewedSharedIds(newViewed);
      localStorage.setItem('viewedSharedIds', JSON.stringify(newViewed));
    }
  };

  const fetchReports = async () => {
    try {
      const res = await axios.get(`${backendUrl}/api/reports`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      setReports(res.data);
      
      // If we are currently on the shared tab, mark new ones as read immediately
      if (activeTab === 'shared_reports' && res.data.shared.length > 0) {
        const allSharedIds = res.data.shared.map(r => r._id);
        const newViewed = [...new Set([...viewedSharedIds, ...allSharedIds])];
        setViewedSharedIds(newViewed);
        localStorage.setItem('viewedSharedIds', JSON.stringify(newViewed));
      }
    } catch (err) {
      console.error('Failed to fetch reports', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [backendUrl, authToken]);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      await axios.post(`${backendUrl}/api/reports/generate`, {}, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      await fetchReports();
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || 'Failed to generate report. Make sure Gemini API key is set.');
    } finally {
      setGenerating(false);
    }
  };

  const handleShare = async (reportId) => {
    if (!shareEmail) return;
    try {
      await axios.post(`${backendUrl}/api/reports/share`, {
        report_id: reportId,
        email: shareEmail
      }, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      alert(`Report successfully shared with ${shareEmail}`);
      setShareEmail('');
      setShareTarget(null);
      fetchReports();
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || 'Failed to share report.');
    }
  };

  const ReportCard = ({ report, isShared }) => {
    let barData = null;
    let doughnutData = null;

    if (report.stats && report.stats.hourly) {
      const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
      const activities = Object.keys(report.stats.totals || {});
      
      const datasets = activities.map(act => ({
        label: act,
        data: labels.map((_, i) => report.stats.hourly[i]?.[act] || 0),
        backgroundColor: ACTIVITY_COLORS[act] || "rgba(100, 100, 100, 0.8)",
      }));

      barData = { labels, datasets };
    }

    if (report.stats && report.stats.totals) {
      const labels = Object.keys(report.stats.totals);
      const data = Object.values(report.stats.totals);
      const backgroundColors = labels.map(act => ACTIVITY_COLORS[act] || "rgba(100, 100, 100, 0.8)");
      
      doughnutData = {
        labels,
        datasets: [{
          data,
          backgroundColor: backgroundColors,
          borderWidth: 1,
          borderColor: 'transparent'
        }]
      };
    }

    const barOptions = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'var(--text-muted)' } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'var(--text-muted)' }, title: { display: true, text: 'Minutes', color: 'var(--text-muted)' } }
      },
      plugins: {
        legend: { labels: { color: 'var(--text-main)' } }
      }
    };

    const doughnutOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: 'var(--text-main)' } }
      }
    };

    return (
      <motion.div className="card" style={{ marginBottom: '20px', textAlign: 'left' }} initial={{opacity:0}} animate={{opacity:1}}>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={18} color="var(--primary-color)" />
          {isShared ? `Shared Report from ${report.sender_name || report.user_id}` : `Your Daily Summary - ${report.date}`}
        </h3>
        
        <div style={{ background: 'var(--input-bg)', padding: '16px', borderRadius: '8px', fontSize: '0.95rem', lineHeight: '1.6', whiteSpace: 'pre-wrap', color: 'var(--text-main)' }}>
          {report.report_text}
        </div>

        {report.stats && (
          <div style={{ marginTop: '20px', padding: '16px', background: 'var(--input-bg)', borderRadius: '8px', border: '1px solid var(--card-border)' }}>
            <h4 style={{ fontSize: '1rem', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--secondary-color)' }}>
              <BarChart3 size={18} /> Activity Timeline
            </h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
              {barData && (
                <div style={{ width: '100%', height: '250px' }}>
                  <Bar data={barData} options={barOptions} />
                </div>
              )}
              {doughnutData && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ flex: '1', height: '200px' }}>
                    <Doughnut data={doughnutData} options={doughnutOptions} />
                  </div>
                  <div style={{ flex: '1', color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
                    <strong>Total Activity Overview:</strong>
                    <p style={{ marginTop: '8px' }}>This chart breaks down your total daily movement into proportional segments.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {!isShared && (
          <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--card-border)' }}>
            {shareTarget === report._id ? (
              <div style={{ display: 'flex', gap: '8px' }}>
                <input 
                  type="email" 
                  placeholder="Enter user email..." 
                  value={shareEmail}
                  onChange={(e) => setShareEmail(e.target.value)}
                  style={{ flex: 1, padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--card-border)', background: 'var(--bg-color)', color: 'var(--text-main)' }}
                />
                <button onClick={() => handleShare(report._id)} style={{ padding: '8px 16px', flex: 'none' }}>
                  Share
                </button>
                <button className="btn-stop" onClick={() => setShareTarget(null)} style={{ padding: '8px 16px', flex: 'none' }}>
                  Cancel
                </button>
              </div>
            ) : (
              <button className="btn-stop" onClick={() => setShareTarget(report._id)} style={{ width: 'auto', padding: '8px 16px' }}>
                <Share2 size={16} /> Share Report
              </button>
            )}
            {report.shared_with && report.shared_with.length > 0 && (
              <div style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Shared with: {report.shared_with.join(', ')}
              </div>
            )}
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', paddingTop: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles size={24} color="var(--primary-color)"/> AI Health Reports
        </h2>
        <button className="btn-start" onClick={handleGenerate} disabled={generating} style={{ width: 'auto' }}>
          {generating ? <Loader2 size={18} className="spin" /> : <FileText size={18} />}
          {generating ? 'Generating via Gemini...' : 'Generate Today\'s Report'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', borderBottom: '1px solid var(--card-border)', paddingBottom: '8px' }}>
        <button 
          onClick={() => handleTabChange('my_reports')}
          style={{
            background: 'transparent',
            border: 'none',
            color: activeTab === 'my_reports' ? 'var(--primary-color)' : 'var(--text-muted)',
            fontWeight: activeTab === 'my_reports' ? 'bold' : 'normal',
            borderBottom: activeTab === 'my_reports' ? '2px solid var(--primary-color)' : 'none',
            padding: '8px 16px',
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          My Reports
        </button>
        <button 
          onClick={() => handleTabChange('shared_reports')}
          style={{
            background: 'transparent',
            border: 'none',
            color: activeTab === 'shared_reports' ? 'var(--primary-color)' : 'var(--text-muted)',
            fontWeight: activeTab === 'shared_reports' ? 'bold' : 'normal',
            borderBottom: activeTab === 'shared_reports' ? '2px solid var(--primary-color)' : 'none',
            padding: '8px 16px',
            cursor: 'pointer',
            fontSize: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            position: 'relative'
          }}
        >
          <UserPlus size={16} /> Shared With Me
          {unreadCount > 0 && activeTab !== 'shared_reports' && (
            <span style={{
              background: 'var(--primary-color)',
              color: 'var(--bg-color)',
              borderRadius: '12px',
              padding: '2px 8px',
              fontSize: '0.75rem',
              fontWeight: 'bold',
              marginLeft: '4px'
            }}>
              {unreadCount}
            </span>
          )}
        </button>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}><Loader2 className="spin" size={32} /></div>
      ) : (
        <>
          {activeTab === 'my_reports' && (
            <>
              {reports.own.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No reports generated yet.</div>
              ) : (
                reports.own.map(r => <ReportCard key={r._id} report={r} isShared={false} />)
              )}
            </>
          )}

          {activeTab === 'shared_reports' && (
            <>
              {reports.shared.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No shared reports received yet.</div>
              ) : (
                reports.shared.map(r => <ReportCard key={r._id} report={r} isShared={true} />)
              )}
            </>
          )}
        </>
      )}
    </div>
  );
};

export default Reports;
