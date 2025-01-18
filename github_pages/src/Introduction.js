import Figure from './Figure';
import overviewFigure from './assets/llm_jp_eval_mm_overview.png';
import './Introduction.css';

const Introduction = () => {
  return (
    <div className='introduction'>
      <h1 className='introduction-title'>Introduction</h1>
      <div className='introduction-content'>
        <span>
          We introduce <b>llm-jp-eval-mm</b>, a toolkit for evaluating multiple
          multimodal tasks related to Japanese language performance in a unified
          environment. The toolkit is a benchmarking platform that integrates
          six existing Japanese multimodal tasks and consistently evaluates
          model outputs across multiple metrics. This paper outlines the design
          of llm-jp-eval-mm for its construction and ongoing development,
          reports the results of evaluating 13 publicly available Japanese and
          multilingual VLMs, and discusses the findings in the light of existing
          research.
        </span>
      </div>
      <Figure
        image={overviewFigure} // eslint-disable-line
        altText='Overview of llm-jp-eval-mm'
        caption={
          <>
            Figure 1: <b>Overview of llm-jp-eval-mm.</b>
          </>
        }
      />
    </div>
  );
};

export default Introduction;
