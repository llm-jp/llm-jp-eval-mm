import './Result.css';

const Result = () => {
  return (
    <div className='result'>
      <h1 className='result-title'>Findings</h1>
      <div className='result-content'>
        <span>In this section, we summarize our key observations.</span>
        {/* Culture-specific Split */}
        <div>
          <h2>Model Scaling</h2>
          As the number of parameters increases, the performance of models
          improves across ... TODO:
        </div>
        {/* Scores on Japanese Heritage */}
        <div>
          <h2>Variation in llm-as-a-judge scores.</h2>
          TODO:
          <h3>BAD behaviour of default metrics for each benchmark.</h3>
          TODO:
        </div>
      </div>
    </div>
  );
};

export default Result;
