import { FaGithub, FaRegFilePdf, FaTwitter } from 'react-icons/fa';

// import { AFFILIATION_COLORS, Author, AuthorProps, LinkButton, LinkButtonProps } from "./molecules";
import { AFFILIATION_COLORS } from './Author';
import Author from './Author';
import { AuthorProps } from './Author';
import LinkButton from './LinkButton';
import { LinkButtonProps } from './LinkButton';

import './PaperMetaData.css';

// Paper title
const TITLE = <>llm-jp-eval-mm</>;
const SUBTITLE = (
  <>Automatic Evaluation Platform for Japanese Visual Language Models</>
);

// Authors of the paper
const AUTHORS = [
  {
    name: 'Koki Maeda',
    affiliation: [1, 4],
    annotation1: '†',
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
  },
  {
    name: 'Issa Sugiura',
    affiliation: [2, 4],
    annotation1: '†',
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
  },
  {
    name: 'Yusuke Oda',
    affiliation: [4],
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
  },
  {
    name: 'Shuhei Kurita',
    affiliation: [3, 4],
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
  },
  {
    name: 'Naoaki Okazaki',
    affiliation: [1, 4],
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
    isLast: true,
  },
];

// Description of annotations
const AFFILIATIONS = [
  'dummy for index 0',
  'Institute of Science Tokyo',
  'Kyoto University',
  'NII',
  'NII LLMC',
];
const ANNOTATION_DESCRIPTION = ['†: Equal Contribution'];

const LINK_BUTTONS = [
  {
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
    children: (
      <>
        <FaRegFilePdf />
        &nbsp;Paper (arXiv)
      </>
    ),
  },
  {
    url: 'https://github.com/llm-jp/llm-jp-eval-mm',
    children: (
      <>
        <FaGithub />
        &nbsp;Code
      </>
    ),
  },
];

const PaperMetadata = () => {
  return (
    <div className='paper-metadata'>
      <h1 className='paper-title'>{TITLE}</h1>
      <h2 className='paper-subtitle'>{SUBTITLE}</h2>
      <div className='authors'>
        {AUTHORS.map((author, index) => (
          <Author key={`author${index}`} {...author} />
        ))}
      </div>
      <div className='authors-affiliations'>
        {AFFILIATIONS.map((affiliation, index) => {
          if (index === 0) return null;
          return (
            <span key={`affiliation${index}`}>
              <span
                style={{ color: AFFILIATION_COLORS[index] }}
                key={`affiliation${index}head`}
              >
                {index}
              </span>
              : {affiliation}
              {index !== AFFILIATIONS.length - 1 && <>,&nbsp;</>}
            </span>
          );
        })}
      </div>
      <div className='annotation-description'>
        {ANNOTATION_DESCRIPTION.map((description, index) => (
          <p key={`description${index}`}>{description}</p>
        ))}
      </div>
      <div className='link-buttons'>
        {LINK_BUTTONS.map((linkButton, index) => (
          <LinkButton key={`linkButton${index}`} url={linkButton.url}>
            {linkButton.children}
          </LinkButton>
        ))}
      </div>
    </div>
  );
};

export default PaperMetadata;
