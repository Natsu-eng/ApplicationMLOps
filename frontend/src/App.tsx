import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { ToastProvider } from "./components/ui/Toast";
import { CommandPalette } from "./components/ui/CommandPalette";
import Aide from "./pages/Aide";
import AllHistory from "./pages/AllHistory";
import AnomalyDetection from "./pages/AnomalyDetection";
import Clustering from "./pages/Clustering";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import DesignSystem from "./pages/DesignSystem";
import DimensionalityReduction from "./pages/DimensionalityReduction";
import Login from "./pages/Login";
import Onboarding from "./pages/Onboarding";
import Orientation from "./pages/Orientation";
import Profile from "./pages/Profile";
import Register from "./pages/Register";
import Training from "./pages/Training";
import TrainingHistory from "./pages/TrainingHistory";
import UnsupervisedHistory from "./pages/UnsupervisedHistory";
import VisionAnomalies from "./pages/VisionAnomalies";
import VisionClassification from "./pages/VisionClassification";
import VisionDatasets from "./pages/VisionDatasets";
import VisionHistory from "./pages/VisionHistory";

/** Palette de commandes globale (⌘K) — n'a de sens que pour un utilisateur
 * connecté (les destinations sont toutes des routes protégées) : montée
 * ici, à l'intérieur de `<Router>` (nécessite `useNavigate`) mais après
 * `AuthProvider`, pour lire `isAuth`. */
function GlobalCommandPalette() {
  const { isAuth } = useAuth();
  if (!isAuth) return null;
  return <CommandPalette />;
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ToastProvider>
        <Router>
          <GlobalCommandPalette />
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
              path="/onboarding"
              element={
                <ProtectedRoute>
                  <Onboarding />
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
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/historique"
              element={
                <ProtectedRoute>
                  <AllHistory />
                </ProtectedRoute>
              }
            />
            <Route
              path="/aide"
              element={
                <ProtectedRoute>
                  <Aide />
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
            <Route
              path="/non-supervise/historique"
              element={
                <ProtectedRoute>
                  <UnsupervisedHistory />
                </ProtectedRoute>
              }
            />
            <Route path="/vision" element={<Navigate to="/vision/classification" replace />} />
            <Route
              path="/vision/datasets"
              element={
                <ProtectedRoute>
                  <VisionDatasets />
                </ProtectedRoute>
              }
            />
            <Route
              path="/vision/classification"
              element={
                <ProtectedRoute>
                  <VisionClassification />
                </ProtectedRoute>
              }
            />
            <Route
              path="/vision/anomalies"
              element={
                <ProtectedRoute>
                  <VisionAnomalies />
                </ProtectedRoute>
              }
            />
            <Route
              path="/vision/historique"
              element={
                <ProtectedRoute>
                  <VisionHistory />
                </ProtectedRoute>
              }
            />
            {/* Lot 2A — page de style guide, protégée mais jamais liée
                depuis la navigation (accès direct par URL uniquement).
                /dev/components (Lot 2, mission refonte visuelle) est un
                alias de la même page — un seul contenu à maintenir, deux
                chemins d'accès (l'historique du projet utilisait déjà
                /design avant que la mission ne nomme explicitement
                /dev/components). */}
            <Route
              path="/design"
              element={
                <ProtectedRoute>
                  <DesignSystem />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dev/components"
              element={
                <ProtectedRoute>
                  <DesignSystem />
                </ProtectedRoute>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
