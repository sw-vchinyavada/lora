import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Plus, Search, Users, FilePlus } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Applicants() {
  const { api } = useAuth();
  const [applicants, setApplicants] = useState([]);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const url = q ? `/applicants?q=${encodeURIComponent(q)}` : "/applicants";
    api(url)
      .then((r) => (r.ok ? r.json() : []))
      .then(setApplicants)
      .catch(() => setApplicants([]))
      .finally(() => setLoading(false));
  }, [api, q]);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-slate-900">Applicants</h1>
          <p className="text-slate-500 mt-1">Manage and score credit applicants</p>
        </div>
        <Link to="/applicants/new" className="btn-primary inline-flex items-center gap-2">
          <Plus size={20} />
          Add Applicant
        </Link>
      </div>

      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search by ID, name, national ID…"
          className="input-field pl-12 w-full md:w-96"
        />
      </div>

      <div className="card overflow-hidden">
        {loading ? (
          <div className="p-12 flex items-center justify-center">
            <div className="animate-pulse flex flex-col items-center gap-3">
              <Users size={40} className="text-slate-300" />
              <div className="h-4 w-32 bg-slate-200 rounded" />
            </div>
          </div>
        ) : applicants.length === 0 ? (
          <div className="p-12 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-100 mb-4">
              <FilePlus size={32} className="text-slate-400" />
            </div>
            <h3 className="font-display font-semibold text-slate-800">No applicants yet</h3>
            <p className="text-slate-500 mt-2 max-w-sm mx-auto">
              Add your first applicant to start running credit scores using the LoRA model.
            </p>
            <Link to="/applicants/new" className="btn-primary inline-flex items-center gap-2 mt-6">
              <Plus size={20} />
              Add Applicant
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Applicant ID</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Name</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">Location</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700">MSME</th>
                  <th className="text-left py-4 px-6 font-display font-semibold text-slate-700"></th>
                </tr>
              </thead>
              <tbody>
                {applicants.map((a) => (
                  <tr key={a.id} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                    <td className="py-4 px-6">
                      <Link
                        to={`/applicants/${a.applicant_id}`}
                        className="font-medium text-primary-600 hover:text-primary-700 hover:underline"
                      >
                        {a.applicant_id}
                      </Link>
                    </td>
                    <td className="py-4 px-6 text-slate-800">{a.full_name || "—"}</td>
                    <td className="py-4 px-6 text-slate-600 capitalize">{a.location || "—"}</td>
                    <td className="py-4 px-6">
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${a.msme ? "bg-primary-50 text-primary-700" : "bg-slate-100 text-slate-600"}`}>
                        {a.msme ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <Link
                        to={`/applicants/${a.applicant_id}`}
                        className="btn-ghost text-sm"
                      >
                        View →
                      </Link>
                    </td>
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
