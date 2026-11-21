export const AFFILIATION_COLORS = [
  "",
  "#6fbf73",
  "#ed4b82",
  "#9400d3",
  "#4169E1",
  "#ffac33",
  "#1e90ff",
  "#ff69b4",
];

export type AuthorProps = {
  name: string;
  affiliation: number[];
  annotation1?: string;
  annotation2?: string;
  url?: string;
  isLast?: boolean;
};

export function Author({
  name,
  affiliation,
  annotation1,
  annotation2,
  url,
  isLast,
}: AuthorProps) {
  const nameNode = url ? (
    <a
      href={url}
      target="_blank"
      rel="noreferrer noopener"
      className="text-foreground underline-offset-4 hover:underline"
    >
      {name}
    </a>
  ) : (
    <span>{name}</span>
  );

  return (
    <span className="inline-flex items-baseline gap-0.5 text-base">
      {nameNode}
      <sup className="text-xs">
        {annotation1}
        {affiliation.map((num, i) => (
          <span key={num}>
            <span style={{ color: AFFILIATION_COLORS[num] }}>{num}</span>
            {i < affiliation.length - 1 && ","}
          </span>
        ))}
        {annotation2}
      </sup>
      {!isLast && <span>,&nbsp;</span>}
    </span>
  );
}
