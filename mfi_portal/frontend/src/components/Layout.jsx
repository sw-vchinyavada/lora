import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { LayoutDashboard, Users, FileBarChart, LogOut, Shield, Sparkles } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const nav = [
    { to: "/", icon: LayoutDashboard, label: "Dashboard" },
    { to: "/applicants", icon: Users, label: "Applicants" },
    { to: "/reports", icon: FileBarChart, label: "Reports" },
  ];

  return (
    <div className="min-h-screen flex bg-slate-50">
      <aside className="w-64 bg-slate-950 text-white flex flex-col shadow-xl">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-glow">
              <Shield size={22} />
            </div>
            <div>
              <h1 className="font-display font-bold text-lg">Credit Portal</h1>
              <p className="text-slate-400 text-xs flex items-center gap-1">
                <Sparkles size={10} /> LoRA Scoring
              </p>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-4 space-y-1">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
                  isActive
                    ? "bg-primary-600 text-white shadow-lg shadow-primary-600/25"
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-white"
                }`
              }
            >
              <Icon size={20} strokeWidth={2} />
              <span className="font-medium">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-slate-800 space-y-1">
          <div className="px-4 py-2 rounded-lg bg-slate-800/50">
            <p className="text-sm font-medium truncate">{user?.full_name || user?.username}</p>
            <p className="text-xs text-slate-400 capitalize">{user?.role}</p>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-slate-400 hover:bg-slate-800/50 hover:text-white transition-colors"
          >
            <LogOut size={20} />
            <span className="font-medium">Sign out</span>
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-8">
        <Outlet />
      </main>
    </div>
  );
}
