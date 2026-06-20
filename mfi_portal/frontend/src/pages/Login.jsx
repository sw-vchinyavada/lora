import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Shield, Lock, User } from "lucide-react";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (user) {
    navigate(location.state?.from || "/", { replace: true });
    return null;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
      navigate(location.state?.from || "/", { replace: true });
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      <div className="hidden lg:flex lg:w-1/2 bg-slate-950 p-12 flex-col justify-between">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-glow">
              <Shield size={28} />
            </div>
            <div>
              <h1 className="font-display font-bold text-xl text-white">Credit Portal</h1>
              <p className="text-slate-400 text-sm">LoRA Scoring</p>
            </div>
          </div>
        </div>
        <div className="space-y-6">
          <h2 className="text-2xl font-display font-bold text-white leading-tight">
            Credit scoring for financial inclusion
          </h2>
          <p className="text-slate-400 max-w-md">
            LoRA-enhanced alternative data credit scoring. Mobile money, utility payments, and demographics — aligned with NDS1/NDS2.
          </p>
          <div className="flex gap-4 pt-4">
            <div className="px-4 py-2 rounded-lg bg-slate-800/50 text-slate-300 text-sm">
              Fairness tested
            </div>
            <div className="px-4 py-2 rounded-lg bg-slate-800/50 text-slate-300 text-sm">
              Explainable AI
            </div>
          </div>
        </div>
      </div>
      <div className="flex-1 flex items-center justify-center p-8 bg-slate-50">
        <div className="w-full max-w-md">
          <div className="lg:hidden mb-8 text-center">
            <div className="inline-flex items-center gap-2">
              <Shield size={32} className="text-primary-600" />
              <span className="font-display font-bold text-xl">Credit Portal</span>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-soft border border-slate-200/80 p-8">
            <h2 className="text-xl font-display font-bold text-slate-900 text-center">Sign in</h2>
            <p className="text-slate-500 text-sm text-center mt-2">Access the credit scoring portal</p>
            <form onSubmit={handleSubmit} className="mt-8 space-y-5">
              {error && (
                <div className="p-4 bg-danger-50 border border-danger-200 text-danger-600 rounded-xl text-sm flex items-center gap-2">
                  <Lock size={18} />
                  {error}
                </div>
              )}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Username</label>
                <div className="relative">
                  <User size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="input-field pl-11"
                    placeholder="Enter username"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
                <div className="relative">
                  <Lock size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="input-field pl-11"
                    placeholder="••••••••"
                    required
                  />
                </div>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="btn-primary w-full py-3"
              >
                {loading ? "Signing in…" : "Sign in"}
              </button>
            </form>
            <p className="mt-6 text-center text-xs text-slate-500">
              Demo: admin/admin123 or officer/officer123
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
