"use client";

import { Copy } from "lucide-react";
import { useState } from "react";

const BIBTEX_ENTRIES = `@article{maeda-etal-2026-evalmm,
  title = {日本語視覚言語モデルのタスク横断評価と実証的分析},
  author = {前田 航希 and
    杉浦 一瑳 and
    小田 悠介 and
    栗田 修平 and
    岡崎 直観},
  journal = {自然言語処理},
  volume = {33},
  number = {2},
  pages = {TBD},
  year = {2026},
  note = {local.ja},
  month = {January}
}

@article{maeda-etal-2026-evalmm-en,
  title = {Cross-Task Evaluation and Empirical Analysis of Japanese Visual Language Models},
  author = {Koki Maeda and
    Issa Sugiura and
    Yusuke Oda and
    Shuhei Kurita and
    Naoaki Okazaki},
  journal = {自然言語処理},
  volume = {33},
  number = {2},
  pages = {TBD},
  year = {2026},
  note = {local.en},
  month = {January}
}
`;

export function BibTex() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX_ENTRIES);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Failed to copy BibTeX entry", e);
    }
  };

  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-2xl font-semibold tracking-tight">BibTeX</h2>
      <div className="relative">
        <pre className="overflow-x-auto rounded-md border bg-muted p-4 text-xs leading-relaxed">
          <code>{BIBTEX_ENTRIES}</code>
        </pre>
        <button
          type="button"
          onClick={handleCopy}
          aria-label="Copy BibTeX"
          className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-md border bg-background px-2 py-1 text-xs transition-colors hover:bg-accent"
        >
          <Copy className="h-3.5 w-3.5" />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </section>
  );
}
