import { useEffect, useState } from "react";
import { Download, FileText, Clock } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Reports() {
  const { api } = useAuth();
  const [activity, setActivity] = useState([]);
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api(`/reports/activity?days=${days}`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setActivity)
      .catch(() => setActivity([]))
      .finally(() => setLoading(false));
  }, [api, days]);

  const exportCsv = async () => {
    const token = localStorage.getItem("token");
    const res = await fetch(`/api/reports/export/csv?days=${days}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `credit_scores_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-slate-900">Reports</h1>
          <p className="text-slate-500 mt-1">Activity log and exports</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={days}
            onChange={(e) => setDays(+e.target.value)}
            className="input-field w-auto"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <button onClick={exportCsv} className="btn-primary flex items-center gap-2">
            <Download size={18} />
            Export CSV
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="card-header flex items-center gap-2">
          <Clock size={20} className="text-primary-600" />
          Recent Activity
        </div>
        {loading ? (
          <div className="p-12 flex items-center justify-center">
            <div className="animate-pulse flex flex-col items-center gap-3">
              <FileText size={40} className="text-slate-300" />
              <div className="h-4 w-32 bg-slate-200 rounded" />
            </div>
          </div>
        ) : activity.length === 0 ? (
          <div className="p-12 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-100 mb-4">
              <FileText size={32} className="text-slate-400" />
            </div>
            <h3 className="font-display font-semibold text-slate-800">No activity in this period</h3>
            <p className="text-slate-500 mt-2">Score inquiries will appear here.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Applicant</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Score</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Risk</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Date</th>
                </tr>
              </thead>
              <tbody>
                {activity.map((a, i) => (
                  <tr key={i} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                    <td className="py-4 px-6">
                      <span className="font-medium text-slate-800">{a.applicant_id}</span>
                      {a.full_name && <span className="text-slate-500 ml-2">({a.full_name})</span>}
                    </td>
                    <td className="py-4 px-6 font-display font-semibold tabular-nums">{a.score}</td>
                    <td className="py-4 px-6">
                      <span
                        className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                          a.risk_band === "Low" ? "bg-success-50 text-success-600" : a.risk_band === "Medium" ? "bg-warning-50 text-warning-600" : "bg-danger-50 text-danger-600"
                        }`}
                      >
                        {a.risk_band}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-slate-500">{new Date(a.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
