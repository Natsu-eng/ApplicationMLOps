import { lazy, Suspense } from "react";
import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import PlatformAdmin from "./pages/PlatformAdmin";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { ToastProvider } from "./components/ui/Toast";
import { CommandPalette } from "./components/ui/CommandPalette";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { RouteFallback } from "./components/RouteFallback";

// Découpage par route (Lot dashboard-dynamique/modernité) — avant ce
// correctif, les 24 pages étaient importées statiquement en tête de
// fichier : un seul chunk JS de 1,15+ Mo minifié (confirmé par
// `npm run build`, avertissement Vite explicite), téléchargé et parsé en
// entier avant même d'afficher l'écran de connexion. `React.lazy` scinde
// chaque page en son propre chunk, chargé à la demande — un visiteur non
// connecté ne télécharge jamais le code des 6 piliers ML, et `/design`
// (page de style guide interne, jamais liée depuis la navigation) ne
// pèse plus jamais sur le chargement initial de personne.
const Aide = lazy(() => import("./pages/Aide"));
const AllHistory = lazy(() => import("./pages/AllHistory"));
const AnomalyDetection = lazy(() => import("./pages/AnomalyDetection"));
const Clustering = lazy(() => import("./pages/Clustering"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Datasets = lazy(() => import("./pages/Datasets"));
const DesignSystem = lazy(() => import("./pages/DesignSystem"));
const DimensionalityReduction = lazy(() => import("./pages/DimensionalityReduction"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword"));
const Login = lazy(() => import("./pages/Login"));
const Onboarding = lazy(() => import("./pages/Onboarding"));
const Orientation = lazy(() => import("./pages/Orientation"));
const Profile = lazy(() => import("./pages/Profile"));
const Register = lazy(() => import("./pages/Register"));
const ResetPassword = lazy(() => import("./pages/ResetPassword"));
const Training = lazy(() => import("./pages/Training"));
const TrainingHistory = lazy(() => import("./pages/TrainingHistory"));
const UnsupervisedHistory = lazy(() => import("./pages/UnsupervisedHistory"));
const VisionAnomalies = lazy(() => import("./pages/VisionAnomalies"));
const VisionClassification = lazy(() => import("./pages/VisionClassification"));
const VisionDatasets = lazy(() => import("./pages/VisionDatasets"));
const VisionHistory = lazy(() => import("./pages/VisionHistory"));

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
          <ErrorBoundary>
          <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPassword />} />
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
            {/* Administration de la plateforme (éditeur). La page se
                contente de ne rien afficher d'incompréhensible à qui n'y a
                pas droit : la protection réelle est côté serveur, où chaque
                route /admin renvoie 403 à un compte ordinaire. */}
            <Route
              path="/admin"
              element={
                <ProtectedRoute>
                  <PlatformAdmin />
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
          </Suspense>
          </ErrorBoundary>
        </Router>
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
