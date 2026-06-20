import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Users, TrendingUp, Activity, ArrowRight, Plus, FileText, BarChart3, Sparkles } from "lucide-react";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { useAuth } from "../context/AuthContext";

export default function Dashboard() {
  const { api } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api("/reports/dashboard")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [api]);

  if (loading) {
    return (
      <div className="animate-pulse space-y-8">
        <div className="h-8 w-48 bg-slate-200 rounded-lg" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card p-6 h-28" />
          ))}
        </div>
      </div>
    );
  }

  const stats = [
    { label: "Total Applicants", value: data?.total_applicants ?? 0, icon: Users, color: "from-primary-500 to-primary-600", bg: "bg-primary-50" },
    { label: "Score Inquiries", value: data?.total_score_inquiries ?? 0, icon: TrendingUp, color: "from-emerald-500 to-emerald-600", bg: "bg-emerald-50" },
    { label: "Today's Scores", value: data?.scores_today ?? 0, icon: Activity, color: "from-amber-500 to-amber-600", bg: "bg-amber-50" },
  ];

  const risk = data?.risk_distribution ?? { low: 0, medium: 0, high: 0 };
  const total = risk.low + risk.medium + risk.high;
  const pieData = [
    { name: "Low", value: risk.low, color: "#10b981" },
    { name: "Medium", value: risk.medium, color: "#f59e0b" },
    { name: "High", value: risk.high, color: "#ef4444" },
  ].filter((d) => d.value > 0);

  const actions = [
    { to: "/applicants/new", icon: Plus, label: "Add new applicant", desc: "Register a new credit applicant" },
    { to: "/applicants", icon: Users, label: "View applicants", desc: "Search and manage applicants" },
    { to: "/reports", icon: FileText, label: "Export reports", desc: "Download CSV reports" },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-slate-900">Dashboard</h1>
          <p className="text-slate-500 mt-1">Overview of credit scoring activity</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map(({ label, value, icon: Icon, color, bg }) => (
          <div
            key={label}
            className="card p-6 hover:shadow-soft transition-shadow duration-200"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-sm font-medium">{label}</p>
                <p className="text-3xl font-display font-bold text-slate-900 mt-1 tabular-nums">{value}</p>
              </div>
              <div className={`p-4 rounded-xl bg-gradient-to-br ${color} text-white shadow-lg`}>
                <Icon size={24} strokeWidth={2} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card overflow-hidden">
          <div className="card-header flex items-center gap-2">
            <BarChart3 size={20} className="text-primary-600" />
            Risk Distribution
          </div>
          <div className="p-6 h-64">
            {total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => [v, ""]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400">
                <div className="text-center">
                  <BarChart3 size={48} className="mx-auto mb-2 opacity-50" />
                  <p>No score data yet</p>
                  <p className="text-sm">Run credit checks to see distribution</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="card-header">Quick Actions</div>
          <div className="p-4 space-y-2">
            {actions.map(({ to, icon: Icon, label, desc }) => (
              <Link
                key={to}
                to={to}
                className="flex items-center gap-4 p-4 rounded-xl hover:bg-slate-50 transition-colors group"
              >
                <div className="p-2.5 rounded-lg bg-slate-100 group-hover:bg-primary-50 text-slate-600 group-hover:text-primary-600 transition-colors">
                  <Icon size={20} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-800 group-hover:text-primary-600 transition-colors">{label}</p>
                  <p className="text-sm text-slate-500">{desc}</p>
                </div>
                <ArrowRight size={18} className="text-slate-300 group-hover:text-primary-500 group-hover:translate-x-1 transition-all" />
              </Link>
            ))}
          </div>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="card-header flex items-center gap-2">
          <Sparkles size={18} className="text-primary-600" />
          Research Context — LoRA Credit Scoring
        </div>
        <div className="p-6 bg-gradient-to-r from-primary-50/50 to-slate-50 border-t border-slate-100">
          <p className="text-slate-600 text-sm leading-relaxed">
            This portal uses a <strong>LoRA</strong> (Low-Rank Adaptation) enhanced credit scoring model trained on Zimbabwe alternative data — mobile money, utility payments, and demographics. 
            Aligned with NDS1/NDS2 financial inclusion goals. Fairness metrics across gender, location, age, and MSME status. 
            <span className="text-slate-500"> — Hu et al. (2021) · RBZ Responsible AI Guidelines</span>
          </p>
        </div>
      </div>
    </div>
  );
}
