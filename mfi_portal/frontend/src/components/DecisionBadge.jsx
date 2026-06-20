/** Approve / Review / Decline recommendation badge. */
export default function DecisionBadge({ riskBand, score }) {
  let decision, bg, text, icon;
  if (riskBand === "Low" || score >= 670) {
    decision = "Approve";
    bg = "bg-success-50";
    text = "text-success-600";
    icon = "✓";
  } else if (riskBand === "High" || score < 580) {
    decision = "Decline";
    bg = "bg-danger-50";
    text = "text-danger-600";
    icon = "✗";
  } else {
    decision = "Review";
    bg = "bg-warning-50";
    text = "text-warning-600";
    icon = "○";
  }
  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg ${bg} ${text} font-semibold`}>
      <span className="text-lg">{icon}</span>
      <span>Recommendation: {decision}</span>
    </div>
  );
}
