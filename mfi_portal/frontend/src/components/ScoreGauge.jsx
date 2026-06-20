/** Credit score gauge (300–850) with risk bands. */
export default function ScoreGauge({ score, riskBand, size = 220 }) {
  const min = 300;
  const max = 850;
  const pct = Math.max(0, Math.min(1, (score - min) / (max - min)));
  const angle = -140 + 280 * pct;

  const riskColors = {
    Low: { stroke: "#10b981", bg: "#ecfdf5" },
    Medium: { stroke: "#f59e0b", bg: "#fffbeb" },
    High: { stroke: "#ef4444", bg: "#fef2f2" },
  };
  const { stroke } = riskColors[riskBand] || riskColors.Medium;

  const r = 75;
  const cx = 100;
  const cy = 100;
  const rad = (angle * Math.PI) / 180;
  const ex = cx + r * Math.cos(rad);
  const ey = cy - r * Math.sin(rad);

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size * 0.7 }}>
        <svg viewBox="0 0 200 140" className="w-full h-full" style={{ overflow: "visible" }}>
          <defs>
            <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="45%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
          <path
            d="M 25 105 A 75 75 0 0 1 175 105"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="16"
            strokeLinecap="round"
          />
          <path
            d="M 25 105 A 75 75 0 0 1 175 105"
            fill="none"
            stroke="url(#arcGrad)"
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={`${235 * pct} 500`}
          />
          <line x1={cx} y1={cy} x2={ex} y2={ey} stroke={stroke} strokeWidth="4" strokeLinecap="round" />
          <circle cx={cx} cy={cy} r="8" fill={stroke} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-4">
          <span className="text-4xl font-display font-bold text-slate-800 tabular-nums">{score}</span>
          <span className="text-xs font-medium text-slate-500 uppercase tracking-wider mt-1">Score</span>
        </div>
      </div>
      <div
        className="mt-3 px-5 py-2 rounded-full text-sm font-semibold"
        style={{
          backgroundColor: riskColors[riskBand]?.bg || "#f8fafc",
          color: stroke,
        }}
      >
        {riskBand} risk
      </div>
    </div>
  );
}
