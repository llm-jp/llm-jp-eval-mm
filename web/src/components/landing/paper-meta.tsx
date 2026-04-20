import { Code2, FileText } from "lucide-react";
import { AFFILIATION_COLORS, Author, type AuthorProps } from "./author";
import { LinkButton } from "./link-button";

const TITLE = "llm-jp-eval-mm";
const SUBTITLE = "Automatic Evaluation Platform for Japanese Visual Language Models";

const AUTHORS: AuthorProps[] = [
  { name: "Koki Maeda", affiliation: [1, 4], annotation1: "†", url: "https://github.com/llm-jp/llm-jp-eval-mm" },
  { name: "Issa Sugiura", affiliation: [2, 4], annotation1: "†", url: "https://github.com/llm-jp/llm-jp-eval-mm" },
  { name: "Yusuke Oda", affiliation: [4], url: "https://github.com/llm-jp/llm-jp-eval-mm" },
  { name: "Shuhei Kurita", affiliation: [3, 4], url: "https://github.com/llm-jp/llm-jp-eval-mm" },
  { name: "Naoaki Okazaki", affiliation: [1, 4], url: "https://github.com/llm-jp/llm-jp-eval-mm", isLast: true },
];

const AFFILIATIONS = [
  "", // index 0 sentinel
  "Institute of Science Tokyo",
  "Kyoto University",
  "NII",
  "NII LLMC",
];

const LINK_BUTTONS = [
  {
    url: "https://github.com/llm-jp/llm-jp-eval-mm",
    label: (
      <>
        <FileText className="h-4 w-4" /> Paper (arXiv)
      </>
    ),
  },
  {
    url: "https://github.com/llm-jp/llm-jp-eval-mm",
    label: (
      <>
        <Code2 className="h-4 w-4" /> Code
      </>
    ),
  },
];

export function PaperMeta() {
  return (
    <section className="flex flex-col items-center gap-4 text-center">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">{TITLE}</h1>
      <h2 className="max-w-3xl text-lg text-muted-foreground sm:text-xl">{SUBTITLE}</h2>

      <div className="flex flex-wrap justify-center gap-x-1 gap-y-2">
        {AUTHORS.map((author) => (
          <Author key={author.name} {...author} />
        ))}
      </div>

      <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
        {AFFILIATIONS.map((name, i) => {
          if (i === 0) return null;
          return (
            <span key={i}>
              <span style={{ color: AFFILIATION_COLORS[i] }}>{i}</span>: {name}
            </span>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground">†: Equal Contribution</p>

      <div className="mt-2 flex flex-wrap justify-center gap-3">
        {LINK_BUTTONS.map((b, i) => (
          <LinkButton key={i} url={b.url}>
            {b.label}
          </LinkButton>
        ))}
      </div>
    </section>
  );
}
