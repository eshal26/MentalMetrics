import { COLORS } from "../utils/constants";

export default function PredictionBadge({ label, confidence, large = false }) {
  const isMdd = label === "Depressed";

  return (
    <div
      className={`prediction-badge-card ${large ? "large" : ""}`}
      style={{
        borderColor: isMdd ? COLORS.danger : COLORS.success,
        color: isMdd ? COLORS.danger : COLORS.success,
      }}
    >
      <span className="prediction-badge-label">{label}</span>
      <span className="prediction-badge-confidence">{confidence.toFixed(1)}%</span>
    </div>
  );
}
