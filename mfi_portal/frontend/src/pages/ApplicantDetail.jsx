import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Calculator, History, User, AlertCircle } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import ScoreGauge from "../components/ScoreGauge";
import DecisionBadge from "../components/DecisionBadge";
import { useToast } from "../components/ToastContext";

export default function ApplicantDetail() {
  const { id } = useParams();
  const { api } = useAuth();
  const toast = useToast();
  const [applicant, setApplicant] = useState(null);
  const [history, setHistory] = useState([]);
  const [scoreResult, setScoreResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scoring, setScoring] = useState(false);

  useEffect(() => {
    api(`/applicants/${id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setApplicant)
      .finally(() => setLoading(false));
  }, [api, id]);

  useEffect(() => {
    if (!id) return;
    api(`/applicants/${id}/history`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setHistory)
      .catch(() => setHistory([]));
  }, [api, id]);

  const runScore = () => {
    setScoring(true);
    setScoreResult(null);
    api(`/applicants/${id}/score`, { method: "POST" })
      .then((r) => {
        if (!r.ok) throw new Error("Scoring failed");
        return r.json();
      })
      .then((data) => {
        setScoreResult(data);
        setHistory((h) => [{ score: data.score, risk_band: data.risk_band, created_at: new Date().toISOString() }, ...h]);
        toast("Credit score computed successfully", "success");
      })
      .catch(() => toast("Scoring model not available. Train models first.", "error"))
      .finally(() => setScoring(false));
  };

  if (loading || !applicant) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="h-6 w-48 bg-slate-200 rounded" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="card h-64 p-6" />
          <div className="card h-64 p-6" />
        </div>
      </div>
    );
  }

  const riskColor = (band) =>
    band === "Low" ? "text-success-600" : band === "Medium" ? "text-warning-600" : "text-danger-600";

  return (
    <div className="space-y-6 animate-fade-in">
      <Link
        to="/applicants"
        className="inline-flex items-center gap-2 text-slate-600 hover:text-slate-900 font-medium transition-colors"
      >
        <ArrowLeft size={18} />
        Back to applicants
      </Link>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-slate-900">Applicant Profile</h1>
          <p className="text-slate-500 mt-1 font-medium">{applicant.applicant_id}</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="card overflow-hidden">
            <div className="card-header flex items-center gap-2">
              <User size={20} className="text-primary-600" />
              Personal & Financial Profile
            </div>
            <div className="p-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                {[
                  ["Full name", applicant.full_name],
                  ["National ID", applicant.national_id],
                  ["Gender", applicant.gender],
                  ["Age", applicant.age],
                  ["Location", applicant.location],
                  ["Region", applicant.region],
                  ["Education", applicant.education],
                  ["Employment", applicant.employment],
                  ["Sector", applicant.sector],
                  ["MSME", applicant.msme ? "Yes" : "No"],
                  ["MM provider", applicant.mm_provider],
                  ["MM txns/mo", applicant.mm_txn_per_month ?? applicant.mm_txn_freq],
                  ["MM tenure (mo)", applicant.mm_tenure_months],
                  ["MM weekend %", applicant.mm_weekend_usage_pct != null ? (applicant.mm_weekend_usage_pct * 100).toFixed(0) + "%" : null],
                  ["Airtime topups/mo", applicant.airtime_topups_per_month],
                  ["Airtime consistency", applicant.airtime_consistency_score != null ? applicant.airtime_consistency_score.toFixed(2) : null],
                  ["Data bundles/mo", applicant.data_bundles_per_month],
                  ["Data avg (USD)", applicant.data_avg_bundle_usd],
                  ["Utility payment rate", applicant.utility_payment_rate != null ? (applicant.utility_payment_rate * 100).toFixed(0) + "%" : null],
                  ["ZESA type", applicant.zesa_type],
                  ["Util consistency", applicant.util_multi_service_consistency != null ? applicant.util_multi_service_consistency.toFixed(2) : null],
                  ["Util overdue", applicant.util_overdue_count],
                  ["Has smartphone", applicant.has_smartphone ? "Yes" : "No"],
                  ["Social media (0–1)", applicant.social_media_usage != null ? applicant.social_media_usage.toFixed(2) : null],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-xs font-medium text-slate-500 uppercase tracking-wider">{k}</dt>
                    <dd className="mt-1 font-medium text-slate-800">{v ?? "—"}</dd>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card overflow-hidden">
            <div className="card-header flex items-center gap-2">
              <History size={20} className="text-primary-600" />
              Score History
            </div>
            <div className="p-6">
              {history.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <History size={40} className="mx-auto mb-3 opacity-50" />
                  <p>No score inquiries yet</p>
                  <p className="text-sm mt-1">Run a credit check above</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((h, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between py-3 px-4 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors"
                    >
                      <span className="font-display font-semibold text-slate-800 tabular-nums">{h.score}</span>
                      <span className={`font-medium ${riskColor(h.risk_band)}`}>{h.risk_band}</span>
                      <span className="text-sm text-slate-500">{new Date(h.created_at).toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card overflow-hidden">
            <div className="card-header flex items-center gap-2">
              <Calculator size={20} className="text-primary-600" />
              Credit Score
            </div>
            <div className="p-6">
              <button
                onClick={runScore}
                disabled={scoring}
                className="btn-primary w-full mb-6"
              >
                {scoring ? "Computing…" : "Check Credit Score"}
              </button>
              {scoreResult ? (
                <div className="animate-slide-up space-y-6">
                  <div className="flex justify-center">
                    <ScoreGauge score={scoreResult.score} riskBand={scoreResult.risk_band} size={200} />
                  </div>
                  <div className="pt-4 border-t border-slate-100 space-y-4">
                    <DecisionBadge riskBand={scoreResult.risk_band} score={scoreResult.score} />
                    <p className="text-xs text-slate-500">LoRA model · Fairness-tested (gender, location, MSME)</p>
                    <div>
                      <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">Default probability</p>
                      <p className="text-lg font-semibold text-slate-800">{(scoreResult.default_probability * 100).toFixed(1)}%</p>
                    </div>
                    {scoreResult.top_drivers?.length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">Top risk factors</p>
                        <ul className="space-y-1.5">
                          {scoreResult.top_drivers.map((d, i) => (
                            <li key={i} className="flex items-center gap-2 text-sm">
                              <span className="w-5 h-5 rounded bg-slate-200 flex items-center justify-center text-xs font-medium text-slate-600">{i + 1}</span>
                              <span className="text-slate-700">{d.feature.replace(/_/g, " ")}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">
                  <AlertCircle size={40} className="mx-auto mb-3 opacity-50" />
                  <p className="text-sm">Click above to run credit score</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
