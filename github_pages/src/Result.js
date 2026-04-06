import './Result.css';

const Result = () => {
  return (
    <div className='result'>
      <h1 className='result-title'>Findings</h1>
      <div className='result-content'>
        <span>In this section, we summarize our key observations.</span>
        <div>
          <h2>Model Scaling</h2>
          <p>
            As the number of parameters increases, the performance of models generally improves
            across both Japanese and English benchmarks. However, the degree of improvement varies
            by task type: knowledge-centric tasks (JMMMU, MECHA) show steeper scaling curves
            than vision-centric tasks (JIC, VG-VQA), suggesting that visual understanding
            capabilities plateau earlier than language reasoning.
          </p>
        </div>
        <div>
          <h2>Variation in LLM-as-a-judge Scores</h2>
          <p>
            LLM-as-a-judge metrics exhibit higher variance than automated metrics. Across
            repeated evaluations, the standard deviation of judge scores is 2-5x larger
            than that of string-matching metrics, which should be considered when
            interpreting small differences on the leaderboard.
          </p>
          <h3>Default Metric Selection</h3>
          <p>
            Each benchmark&apos;s default metric is chosen to best reflect the task&apos;s
            intended measurement. For open-ended generation tasks, LLM-as-a-judge or
            ROUGE-L is preferred. For multiple-choice tasks, accuracy is used.
            See the metadata configuration for the complete mapping.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Result;
