## CHAPTER 7: IMPLEMENTATION (TOOLS AND TECHNOLOGY USED)

### 7.1 Overview of Technology Stack

The system is implemented as a full-stack web application with clearly delineated frontend, backend, machine learning, and data infrastructure layers. Each technology was selected based on specific technical requirements including real-time performance, cross-platform compatibility, scalability, and developer productivity.

**Table 7.1: Tools and Technologies Used**

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Frontend Framework | React | 19.2.5 | Component-based architecture; efficient re-rendering for real-time data |
| Build Tool | Vite | 8.0.9 | Fast HMR; native ES module support; LAN hosting via --host |
| Charting | Chart.js + react-chartjs-2 | 4.5.1 / 5.3.1 | Canvas-based rendering for high-frequency sensor data visualization |
| Animation | Framer Motion | 12.38.0 | Declarative animation API for smooth UI transitions |
| Icons | Lucide React | 1.8.0 | Lightweight, tree-shakeable SVG icon library |
| HTTP Client | Axios | 1.15.1 | Promise-based HTTP with interceptor support |
| Authentication | @react-oauth/google | 0.13.5 | Google OAuth 2.0 implicit grant flow for React |
| Backend Framework | Flask | Latest | Lightweight WSGI framework; suitable for ML model serving |
| CORS | Flask-CORS | Latest | Cross-origin resource sharing for frontend-backend communication |
| ML Framework | TensorFlow / Keras | Latest | Industry-standard deep learning; Keras Sequential API |
| Signal Processing | SciPy | Latest | Butterworth filter design and application |
| Data Processing | NumPy, Pandas | Latest | Efficient numerical computation and tabular data manipulation |
| Preprocessing | scikit-learn | Latest | StandardScaler, LabelEncoder, train_test_split, metrics |
| Database | MongoDB Atlas | Cloud | Document-oriented NoSQL; flexible schema for activity logs |
| Database Driver | PyMongo | Latest | Official MongoDB driver for Python |
| Authentication | PyJWT | Latest | JWT token creation and verification |
| Auth Verification | google-auth | Latest | Server-side Google OAuth token verification |
| AI Integration | google-generativeai | Latest | Gemini 1.5 Flash API client |
| Environment | python-dotenv | Latest | Environment variable management from .env files |
| Deployment | Vercel | Cloud | Frontend hosting with custom header configuration |
| Language | Python 3.12 | 3.12 | Backend and ML pipeline |
| Language | JavaScript (ES2022) | ES2022 | Frontend application logic |

### 7.2 Frontend Implementation Details

#### 7.2.1 Application Entry Point

The application is bootstrapped in main.jsx, which wraps the root App component inside React's StrictMode and the GoogleOAuthProvider from @react-oauth/google. The GoogleOAuthProvider receives the Google Client ID from the VITE_GOOGLE_CLIENT_ID environment variable, enabling Google Sign-In functionality throughout the component tree.

```javascript
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GoogleOAuthProvider clientId={clientId}>
      <App />
    </GoogleOAuthProvider>
  </StrictMode>
)
```

#### 7.2.2 Sensor Data Acquisition Hook (useAccelerometer.js)

The useAccelerometer custom React hook encapsulates the entire sensor data acquisition pipeline. It manages sensor event listeners, data buffering, and window dispatch through the following mechanism:

**Initialization:** Upon calling startCapture(), the hook requests DeviceMotionEvent permission (required on iOS 13+), registers a devicemotion event listener, and starts a setInterval timer at 50ms intervals (20 Hz).

**Data Collection:** The handleMotion callback processes each DeviceMotion event, extracting accelerationIncludingGravity (x, y, z) in m/s squared and rotationRate (alpha, beta, gamma) in degrees per second. Gyroscope values are converted from degrees to radians per second by multiplying with pi/180, as the neural network was trained on radian-unit gyroscope data.

**Buffering and Dispatch:** The tick function executes every 50ms, reads the latest sensor values from the sensorRef, appends a six-element array [ax, ay, az, gx, gy, gz] to the buffer, and checks if the buffer has reached the FRAME_SIZE of 60 samples. Upon reaching capacity, the buffer contents are copied, the buffer is reset, and the onWindowReady callback is invoked with the complete 60x6 window, triggering an HTTP POST to the backend's /predict endpoint.

**Simulation Mode:** When the DeviceMotion API is unavailable (desktop browsers), the hook automatically falls back to generating synthetic sensor data simulating a walking pattern using sinusoidal functions with Gaussian noise, enabling development and testing without a physical mobile device.

#### 7.2.3 Dashboard Layout (App.jsx)

The App component implements a simple client-side routing mechanism using React state (currentView) to switch between the dashboard and reports views. Authentication state (authToken, user) is persisted in localStorage to survive page refreshes.

The dashboard employs a responsive 12-column CSS grid layout with six card components:
- **Status Bar:** Displays connection status with color-coded indicators (green for connected, red for error).
- **Activity Card:** Shows the current predicted activity with animated label transitions using Framer Motion's AnimatePresence, a confidence bar with color gradients (green above 80 percent, blue above 60 percent, red below), and a status badge indicating prediction confidence level.
- **Buffer Status Card:** Displays windowing progress as a progress bar showing samples collected versus the required sixty.
- **Sensor Values Card:** Real-time display of all six sensor axes with formatted numerical values and unit labels.
- **Probabilities Card:** Top-five class probabilities rendered as animated horizontal bars with percentage labels.
- **Sensor Chart Card:** Real-time line chart rendering the last sixty accelerometer samples across three axes using Chart.js.

#### 7.2.4 Sensor Visualization (SensorChart.jsx)

The SensorChart component maintains a rolling buffer of sixty data points for each accelerometer axis (X, Y, Z) using a useRef to persist data across renders without triggering re-renders. On each data update, new values are pushed to the datasets and the oldest values are shifted out. The Chart.js instance is updated using the 'none' animation mode to prevent performance degradation from continuous re-rendering at 20 Hz.

#### 7.2.5 Authentication Component (Login.jsx)

The Login component renders a centered card with a Google Sign-In button. It uses the useGoogleLogin hook from @react-oauth/google configured for the implicit grant flow, which returns an access_token rather than an ID token. This approach was specifically chosen to circumvent Cross-Origin-Opener-Policy (COOP) errors that occur with the standard popup-based ID token flow in certain browser configurations.

#### 7.2.6 Reports Component (Reports.jsx)

The Reports component provides a complete interface for AI health report management:
- A "Generate Today's Report" button that sends an authenticated POST request to /api/reports/generate and displays a loading spinner during Gemini processing.
- A list of the user's own reports rendered as cards with the report date, generated text content, and a share button.
- An inline sharing interface that expands to reveal an email input field for specifying the target user.
- A "Shared With Me" section displaying reports shared by other users.

#### 7.2.7 Design System (index.css)

The frontend employs a professional dark theme design system implemented through CSS custom properties:

- **Color Palette:** Dark navy background (#0f172a), slate card backgrounds (#1e293b), blue primary accent (#3b82f6), emerald success (#10b981), and red warning (#ef4444).
- **Typography:** Inter font family from Google Fonts with weights ranging from 300 (light) to 700 (bold).
- **Card System:** 12px border radius, 1px solid borders, subtle box shadows with hover elevation effects.
- **Responsive Breakpoints:** At 860px viewport width, the 12-column grid collapses to single-column layout.

#### 7.2.8 Deployment Configuration

The Vite development server is configured with host: true to bind to all network interfaces (0.0.0.0), enabling access from mobile devices on the same local network via the host machine's LAN IP address. The vercel.json configuration sets Cross-Origin-Opener-Policy to "same-origin-allow-popups" and Cross-Origin-Embedder-Policy to "unsafe-none" to ensure compatibility with Google OAuth popup authentication on the Vercel deployment platform.

### 7.3 Backend Implementation Details

#### 7.3.1 Application Initialization

The Flask application initializes by loading environment variables from a .env file, establishing a MongoDB Atlas connection, configuring the Gemini AI client, and loading four persisted model artifacts from disk: the Keras model file (har_model.keras), the label encoder (label_encoder.pkl), the activity names mapping (activity_names.pkl), and the StandardScaler (scaler.pkl). Butterworth filter coefficients are precomputed at startup, and filter state arrays are initialized to zeros for each of the six sensor channels (noise filter) and three accelerometer channels (gravity filter).

#### 7.3.2 Real-Time Signal Processing

The real_time_preprocess function implements causal (forward-only) Butterworth filtering using scipy.signal.sosfilt with persistent filter states. Unlike the training-time preprocessing which uses zero-phase bidirectional filtering (sosfiltfilt), real-time inference requires causal filtering since future samples are not yet available. The filter states are maintained as module-level variables, ensuring continuity across consecutive prediction requests within the same server session. This design enables the filtering to operate as if processing a continuous sensor stream rather than isolated windows.

#### 7.3.3 Prediction Endpoint Architecture

The /predict endpoint processes incoming requests through a six-stage pipeline: input validation, signal preprocessing, stationary detection, model inference, confidence thresholding, and majority voting. The stationary detection stage computes the sum of per-axis variances for both accelerometer and gyroscope channels from the raw (pre-filtered) data, bypassing the neural network entirely when both variances fall below 0.15. This heuristic significantly reduces false positive activity classifications when the device is resting on a stationary surface, where minor sensor noise could otherwise be misinterpreted as subtle hand movements.

#### 7.3.4 Authentication Middleware

The token_required decorator implements JWT-based authentication by extracting the Bearer token from the Authorization header, decoding it using the HS256 algorithm with the server's JWT secret, and retrieving the corresponding user document from MongoDB. The decorator injects the authenticated user object as the first argument to the wrapped route handler, enabling clean separation of authentication logic from business logic.

#### 7.3.5 Database Schema Design

**Table 7.4: MongoDB Collections and Document Structure**

| Collection | Field | Type | Description |
|-----------|-------|------|-------------|
| **users** | email | String | Unique identifier from Google |
| | name | String | Display name from Google profile |
| | created_at | DateTime | Account creation timestamp |
| **activity_logs** | user_id | ObjectId | Reference to users collection |
| | timestamp | DateTime | UTC prediction timestamp |
| | activity | String | Smoothed activity label |
| | confidence | Float | Model confidence score |
| **reports** | user_id | ObjectId | Reference to users collection |
| | date | DateTime | Report date (truncated to midnight) |
| | report_text | String | Gemini-generated report content |
| | shared_with | Array[String] | Email addresses of shared users |

MongoDB's document-oriented model was selected for its schema flexibility, which accommodates evolving data requirements without schema migration overhead. The activity_logs collection stores one document per prediction, enabling fine-grained temporal analysis. Reports follow upsert semantics with a compound key of (user_id, date) ensuring one report per user per day.

**Table 7.2: Python Backend Dependencies**

| Package | Purpose |
|---------|---------|
| flask | Web application framework |
| flask-cors | Cross-origin resource sharing |
| tensorflow | Deep learning runtime |
| keras | High-level neural network API |
| numpy | Numerical array operations |
| pandas | Data manipulation (used in data pipeline) |
| scikit-learn | Preprocessing and evaluation metrics |
| python-dotenv | Environment variable loading |
| pymongo | MongoDB driver |
| google-auth, google-auth-oauthlib | OAuth token verification |
| google-generativeai | Gemini API client |
| PyJWT | JWT token management |
| scipy | Signal processing (Butterworth filters) |

**Table 7.3: Frontend Dependencies**

| Package | Purpose |
|---------|---------|
| react, react-dom | UI framework |
| @react-oauth/google | Google Sign-In integration |
| axios | HTTP client for API communication |
| chart.js, react-chartjs-2 | Real-time sensor data charting |
| framer-motion | Animation library |
| lucide-react | Icon components |
| vite, @vitejs/plugin-react | Build tooling |
