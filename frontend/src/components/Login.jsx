import React from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';
import axios from 'axios';

const Login = ({ backendUrl, setAuthToken, setUser }) => {
  const handleSuccess = async (credentialResponse) => {
    try {
      const res = await axios.post(`${backendUrl}/api/auth/google`, {
        token: credentialResponse.credential
      });
      const { token, user } = res.data;
      setAuthToken(token);
      setUser(user);
      localStorage.setItem('har_auth_token', token);
      localStorage.setItem('har_user', JSON.stringify(user));
    } catch (err) {
      console.error('Login failed:', err);
      alert('Login failed. Ensure backend is running and Google Client ID is configured.');
    }
  };

  const handleError = () => {
    console.error('Google Login Failed');
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
      <motion.div 
        className="card" 
        style={{ maxWidth: '400px', width: '100%', textAlign: 'center', padding: '40px 24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Activity size={48} color="var(--primary-color)" style={{ marginBottom: '16px' }} />
        <h2 style={{ justifyContent: 'center', fontSize: '1.5rem', marginBottom: '8px' }}>Sign In</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '32px', fontSize: '0.9rem' }}>
          Authenticate to track your daily activities and generate AI health reports.
        </p>
        
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <GoogleLogin
            onSuccess={handleSuccess}
            onError={handleError}
            useOneTap
            theme="filled_black"
            shape="rectangular"
            ux_mode="popup" 
          />
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
