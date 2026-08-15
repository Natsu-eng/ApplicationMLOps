import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import { AuthProvider } from "./contexts/AuthContext";
import AnomalyDetection from "./pages/AnomalyDetection";
import Clustering from "./pages/Clustering";
import ComingSoon from "./pages/ComingSoon";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import DimensionalityReduction from "./pages/DimensionalityReduction";
import Login from "./pages/Login";
import Orientation from "./pages/Orientation";
import Register from "./pages/Register";
import Training from "./pages/Training";
import TrainingHistory from "./pages/TrainingHistory";

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Orientation />
              </ProtectedRoute>
            }
          />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/datasets"
            element={
              <ProtectedRoute>
                <Datasets />
              </ProtectedRoute>
            }
          />
          <Route
            path="/training"
            element={
              <ProtectedRoute>
                <Training />
              </ProtectedRoute>
            }
          />
          <Route
            path="/training/history"
            element={
              <ProtectedRoute>
                <TrainingHistory />
              </ProtectedRoute>
            }
          />
          <Route
            path="/clustering"
            element={
              <ProtectedRoute>
                <Clustering />
              </ProtectedRoute>
            }
          />
          <Route
            path="/reduction-dimension"
            element={
              <ProtectedRoute>
                <DimensionalityReduction />
              </ProtectedRoute>
            }
          />
          <Route
            path="/anomalies"
            element={
              <ProtectedRoute>
                <AnomalyDetection />
              </ProtectedRoute>
            }
          />
          {/* Pilier réservé — non actif, page de présentation uniquement (voir config/pillars.ts) */}
          <Route
            path="/vision"
            element={
              <ProtectedRoute>
                <ComingSoon pillarId="vision" />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}
