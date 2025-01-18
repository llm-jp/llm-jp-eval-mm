import { LuCopy } from 'react-icons/lu';
import './BibTex.css';

// BibTeX entry for the paper
const BIBTEX_ENTRY = `@inproceedings{maeda2025llm-jp-eval-mm,
author = {前田 航希 and 杉浦 一瑳 and 小田 悠介 and 栗田 修平 and 岡崎 直観},
month = mar,
series = {言語処理学会第31回年次大会 (NLP2025)},
title = {{llm-jp-eval-mm: 日本語視覚言語モデルの自動評価基盤}},
year = {2025}
}
`;

const copyClipboard = () => {
  navigator.clipboard.writeText(BIBTEX_ENTRY).catch((error) => {
    console.error('Failed to copy BibTeX entry to clipboard', error);
  });
};

const BibTeX = () => {
  return (
    <div className='bibtex'>
      <h1 className='bibtex-title'>BibTeX</h1>
      <pre className='bibtex-entry'>
        <code>{BIBTEX_ENTRY}</code>
        <button className='bibtex-copy-button' onClick={copyClipboard}>
          <LuCopy />
        </button>
      </pre>
    </div>
  );
};

export default BibTeX;
