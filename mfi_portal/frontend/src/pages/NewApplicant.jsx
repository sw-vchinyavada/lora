import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { ArrowLeft, User, Smartphone } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../components/ToastContext";

const ZIMBABWE_REGIONS = ["Harare", "Bulawayo", "Midlands", "Mashonaland_East", "Mashonaland_West", "Mashonaland_Central", "Manicaland", "Masvingo", "Matabeleland_North", "Matabeleland_South"];
const initial = {
  applicant_id: "",
  full_name: "",
  national_id: "",
  gender: "male",
  age: 35,
  location: "urban",
  region: "Harare",
  employment: "informal",
  sector: "other",
  msme: 0,
  education: "secondary",
  mm_provider: "ecocash",
  mm_txn_freq: 20,
  mm_txn_per_month: 20,
  mm_avg_amount_usd: 25,
  mm_balance_volatility: 0.5,
  mm_tenure_months: 24,
  mm_weekend_usage_pct: 0.25,
  mm_p2p_ratio: 0.5,
  mm_bill_payment_ratio: 0.3,
  mm_merchant_ratio: 0.2,
  airtime_topups_per_month: 8,
  airtime_avg_amount_usd: 3,
  airtime_consistency_score: 0.6,
  data_bundles_per_month: 3,
  data_avg_bundle_usd: 2.5,
  utility_payment_rate: 0.8,
  zesa_type: "prepaid",
  has_smartphone: 1,
  social_media_usage: 0.5,
  util_electricity_consistency: 0.7,
  util_water_consistency: 0.7,
  util_telecom_consistency: 0.7,
  util_overdue_count: 0,
  util_avg_delay_days: 0,
  util_multi_service_consistency: 0.7,
  consecutive_on_time: 6,
  account_age_months: 12,
  digital_engagement: 0.5,
  channel_diversity: 2,
};

export default function NewApplicant() {
  const { api } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const [form, setForm] = useState(initial);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    const payload = { ...form };
    Object.keys(payload).forEach((k) => {
      if (payload[k] === "" || (typeof payload[k] === "number" && isNaN(payload[k]))) payload[k] = null;
    });
    api("/applicants", { method: "POST", body: JSON.stringify(payload) })
      .then((r) => {
        if (!r.ok) return r.json().then((d) => { throw new Error(d.detail || "Failed"); });
        return r.json();
      })
      .then((data) => {
        toast("Applicant created successfully", "success");
        navigate(`/applicants/${data.applicant_id}`);
      })
      .catch((err) => {
        setError(err.message);
        toast(err.message, "error");
      })
      .finally(() => setLoading(false));
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Link to="/applicants" className="inline-flex items-center gap-2 text-slate-600 hover:text-slate-900 font-medium transition-colors">
        <ArrowLeft size={18} />
        Back
      </Link>
      <h1 className="text-2xl font-display font-bold text-slate-900">New Applicant</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {error && (
          <div className="p-4 bg-danger-50 border border-danger-200 text-danger-600 rounded-xl text-sm">{error}</div>
        )}

        <div className="card overflow-hidden">
          <div className="card-header flex items-center gap-2">
            <User size={20} className="text-primary-600" />
            Identification
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-slate-700 mb-2">Applicant ID *</label>
              <input value={form.applicant_id} onChange={(e) => update("applicant_id", e.target.value)} className="input-field" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Full name</label>
              <input value={form.full_name} onChange={(e) => update("full_name", e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">National ID</label>
              <input value={form.national_id} onChange={(e) => update("national_id", e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Gender</label>
              <select value={form.gender} onChange={(e) => update("gender", e.target.value)} className="input-field">
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Age</label>
              <input type="number" value={form.age} onChange={(e) => update("age", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Location</label>
              <select value={form.location} onChange={(e) => update("location", e.target.value)} className="input-field">
                <option value="urban">Urban</option>
                <option value="rural">Rural</option>
                <option value="peri_urban">Peri-urban</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Region</label>
              <select value={form.region} onChange={(e) => update("region", e.target.value)} className="input-field">
                {ZIMBABWE_REGIONS.map((r) => (
                  <option key={r} value={r}>{r.replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Education</label>
              <select value={form.education} onChange={(e) => update("education", e.target.value)} className="input-field">
                <option value="none">None</option>
                <option value="primary">Primary</option>
                <option value="secondary">Secondary</option>
                <option value="tertiary">Tertiary</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Employment</label>
              <select value={form.employment} onChange={(e) => update("employment", e.target.value)} className="input-field">
                <option value="formal">Formal</option>
                <option value="informal">Informal</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MSME</label>
              <select value={form.msme} onChange={(e) => update("msme", +e.target.value)} className="input-field">
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="card-header flex items-center gap-2">
            <Smartphone size={20} className="text-primary-600" />
            Mobile Money (EcoCash / OneMoney)
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MM provider</label>
              <select value={form.mm_provider} onChange={(e) => update("mm_provider", e.target.value)} className="input-field">
                <option value="ecocash">EcoCash</option>
                <option value="onemoney">OneMoney</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MM txns per month</label>
              <input type="number" value={form.mm_txn_per_month ?? form.mm_txn_freq} onChange={(e) => { update("mm_txn_per_month", +e.target.value); update("mm_txn_freq", +e.target.value); }} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MM avg amount (USD)</label>
              <input type="number" step="0.1" value={form.mm_avg_amount_usd} onChange={(e) => update("mm_avg_amount_usd", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MM tenure (months)</label>
              <input type="number" value={form.mm_tenure_months} onChange={(e) => update("mm_tenure_months", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">MM weekend usage (0–1)</label>
              <input type="number" step="0.01" min="0" max="1" value={form.mm_weekend_usage_pct} onChange={(e) => update("mm_weekend_usage_pct", +e.target.value)} className="input-field" />
            </div>
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="card-header flex items-center gap-2">
            <Smartphone size={20} className="text-primary-600" />
            Telecom (Airtime & Data)
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Airtime topups/month</label>
              <input type="number" value={form.airtime_topups_per_month} onChange={(e) => update("airtime_topups_per_month", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Airtime avg (USD)</label>
              <input type="number" step="0.1" value={form.airtime_avg_amount_usd} onChange={(e) => update("airtime_avg_amount_usd", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Airtime consistency (0–1)</label>
              <input type="number" step="0.01" min="0" max="1" value={form.airtime_consistency_score} onChange={(e) => update("airtime_consistency_score", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Data bundles/month</label>
              <input type="number" value={form.data_bundles_per_month} onChange={(e) => update("data_bundles_per_month", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Data avg bundle (USD)</label>
              <input type="number" step="0.1" value={form.data_avg_bundle_usd} onChange={(e) => update("data_avg_bundle_usd", +e.target.value)} className="input-field" />
            </div>
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="card-header flex items-center gap-2">
            <Smartphone size={20} className="text-primary-600" />
            Utility (ZESA, Water) & Digital
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">ZESA type</label>
              <select value={form.zesa_type} onChange={(e) => update("zesa_type", e.target.value)} className="input-field">
                <option value="prepaid">Prepaid</option>
                <option value="postpaid">Postpaid</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Utility payment rate (0–1)</label>
              <input type="number" step="0.01" min="0" max="1" value={form.utility_payment_rate} onChange={(e) => update("utility_payment_rate", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Util consistency (0–1)</label>
              <input type="number" step="0.01" min="0" max="1" value={form.util_multi_service_consistency} onChange={(e) => update("util_multi_service_consistency", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Util overdue count</label>
              <input type="number" value={form.util_overdue_count} onChange={(e) => update("util_overdue_count", +e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Has smartphone</label>
              <select value={form.has_smartphone} onChange={(e) => update("has_smartphone", +e.target.value)} className="input-field">
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Social media usage (0–1)</label>
              <input type="number" step="0.01" min="0" max="1" value={form.social_media_usage} onChange={(e) => update("social_media_usage", +e.target.value)} className="input-field" />
            </div>
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn-primary px-8 py-3">
          {loading ? "Saving…" : "Create Applicant"}
        </button>
      </form>
    </div>
  );
}
