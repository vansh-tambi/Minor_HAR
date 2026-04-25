import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { FileText, Share2, Loader2, Sparkles, UserPlus } from 'lucide-react';

const Reports = ({ backendUrl, authToken }) => {
  const [reports, setReports] = useState({ own: [], shared: [] });
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [shareEmail, setShareEmail] = useState('');
  const [shareTarget, setShareTarget] = useState(null);

  const fetchReports = async () => {
    try {
      const res = await axios.get(`${backendUrl}/api/reports`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      setReports(res.data);
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

  const ReportCard = ({ report, isShared }) => (
    <motion.div className="card" style={{ marginBottom: '20px', textAlign: 'left' }} initial={{opacity:0}} animate={{opacity:1}}>
      <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <FileText size={18} color="var(--primary-color)" />
        {isShared ? `Shared Report from ${report.user_id}` : `Your Daily Summary - ${report.date}`}
      </h3>
      
      <div style={{ background: 'var(--input-bg)', padding: '16px', borderRadius: '8px', fontSize: '0.95rem', lineHeight: '1.6', whiteSpace: 'pre-wrap', color: 'var(--text-main)' }}>
        {report.report_text}
      </div>
      
      {!isShared && (
        <div style={{ marginTop: '16px' }}>
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

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}><Loader2 className="spin" size={32} /></div>
      ) : (
        <>
          <h3 style={{ marginBottom: '16px', color: 'var(--secondary-color)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>My Reports</h3>
          {reports.own.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No reports generated yet.</div>
          ) : (
            reports.own.map(r => <ReportCard key={r._id} report={r} isShared={false} />)
          )}

          {reports.shared.length > 0 && (
            <>
              <h3 style={{ marginTop: '40px', marginBottom: '16px', color: 'var(--secondary-color)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <UserPlus size={16} /> Shared With Me
              </h3>
              {reports.shared.map(r => <ReportCard key={r._id} report={r} isShared={true} />)}
            </>
          )}
        </>
      )}
    </div>
  );
};

export default Reports;
