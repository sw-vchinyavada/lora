import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { ToastProvider } from "./components/ToastContext";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Applicants from "./pages/Applicants";
import ApplicantDetail from "./pages/ApplicantDetail";
import NewApplicant from "./pages/NewApplicant";
import Reports from "./pages/Reports";

function ProtectedRoute({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="applicants" element={<Applicants />} />
          <Route path="applicants/new" element={<NewApplicant />} />
          <Route path="applicants/:id" element={<ApplicantDetail />} />
          <Route path="reports" element={<Reports />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </ToastProvider>
    </AuthProvider>
  );
}
