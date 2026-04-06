import Figure from './Figure';
import './Method.css';

const Method = () => {
  return (
    <div className='method'>
      <h1 className='method-title'>Design of llm-jp-eval-mm</h1>
      <div className='method-content'>
        <p>
          llm-jp-eval-mm is a lightweight evaluation framework for visual-language models,
          with a strong focus on Japanese multimodal benchmarks. The framework supports
          20+ evaluation tasks across Japanese and English, 15+ scoring metrics, and 50+ model adapters.
        </p>

        <h2>Architecture</h2>
        <p>
          The framework separates concerns into three layers:
        </p>
        <ul>
          <li><strong>Tasks</strong>: Dataset loading, prompt formatting, and answer extraction via a decorator-based registry.</li>
          <li><strong>Metrics</strong>: Scoring and aggregation, including LLM-as-a-judge, string matching, and task-specific scorers.</li>
          <li><strong>Models</strong>: Pluggable inference backends (HuggingFace transformers, vLLM) with per-model adapters.</li>
        </ul>

        <h2>Evaluation Flow</h2>
        <p>
          The CLI (<code>eval-mm run</code>) orchestrates: task loading &rarr; prediction generation &rarr;
          multi-metric scoring &rarr; result persistence. Results are saved in a standardized schema
          consumed by both CUI summaries and GUI visualizations.
        </p>

        <h2>GUI Components</h2>
        <ul>
          <li><strong>Leaderboard</strong> (this page): Public-facing model ranking with task clustering and metric selection.</li>
          <li><strong>Prediction Browser</strong> (Streamlit): Local drill-down into per-sample predictions, multi-image display, and model comparison.</li>
        </ul>
      </div>
    </div>
  );
};

export default Method;
