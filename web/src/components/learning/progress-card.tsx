type ProgressCardProps = {
  detail: string;
  label: string;
  value: number;
};

export function ProgressCard({ detail, label, value }: ProgressCardProps) {
  return (
    <article className="progress-card">
      <div className="progress-copy">
        <h3>{label}</h3>
        <strong>{value}%</strong>
      </div>
      <progress max="100" value={value}>
        {value}%
      </progress>
      <p>{detail}</p>
    </article>
  );
}
