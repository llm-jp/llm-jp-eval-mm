import React, { useEffect, useState } from 'react';
import { FaGithub, FaRegFilePdf } from 'react-icons/fa';
import './Leaderboard.css';
import LinkButton from './LinkButton';

function Leaderboard() {
  const [data, setData] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetUrl, setDatasetUrl] = useState({});

  const [metrics, setMetrics] = useState({});
  const [defaultMetrics, setDefaultMetrics] = useState({});
  const [sortConfig, setSortConfig] = useState(null);

  useEffect(() => {
    // Fetch leaderboard data
    fetch('leaderboard.json')
      .then((response) => response.json())
      .then((data) => {
        setData(data);

        // Extract datasets and metrics dynamically
        const datasetNames = data
          .reduce((acc, row) => {
            return acc.concat(Object.keys(row.scores));
          }, [])
          .filter((value, index, self) => self.indexOf(value) === index);
        setDatasets(datasetNames);

        const metricNames = {};
        datasetNames.forEach((dataset) => {
          const metricNamesArray = data
            .reduce((acc, row) => {
              return acc.concat(Object.keys(row.scores[dataset] || {}));
            }, [])
            .filter((value, index, self) => self.indexOf(value) === index);
          metricNames[dataset] = metricNamesArray;
        });
        setMetrics(metricNames);
      })
      .catch((error) =>
        console.error('Error loading leaderboard data:', error),
      );

    // Fetch default metrics
    fetch('default_metrics.json')
      .then((response) => response.json())
      .then((defaultMetrics) => {
        setDefaultMetrics(defaultMetrics.default_metrics); // Use the `default_metrics` field
      })
      .catch((error) => console.error('Error loading default metrics:', error));

    // Fetch dataset url
    // {
    // {
    //     "Heron": {
    //         "url": "https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench"
    //     },
    //     "JVB-ItW": {
    //         "url": "https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild"
    //     },
    //     "VGVQA": {
    //         "url": "https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500"
    //     },
    //     "MulIm-VQA": {
    //         "url": "https://huggingface.co/datasets/SakanaAI/JA-Multi-Image-VQA"
    //     },
    //     "JDocQA": {
    //         "url": "https://huggingface.co/datasets/shunk031/JDocQA"
    //     },
    //     "JMMMU": {
    //         "url": "https://huggingface.co/datasets/JMMMU/JMMMU"
    //     }
    // }
    fetch('dataset_url.json')
      .then((response) => response.json())
      .then((datasetUrl) => {
        setDatasetUrl(datasetUrl);
      });
  }, []);

  const handleSort = (dataset, metric) => {
    let sortedData = [...data];
    const direction =
      sortConfig?.key === `${dataset}-${metric}` &&
      sortConfig.direction === 'asc'
        ? 'desc'
        : 'asc';
    sortedData.sort((a, b) => {
      const aValue = a.scores[dataset]?.[metric] || 0;
      const bValue = b.scores[dataset]?.[metric] || 0;
      if (aValue < bValue) return direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return direction === 'asc' ? 1 : -1;
      return 0;
    });
    setSortConfig({ key: `${dataset}-${metric}`, direction });
    setData(sortedData);
  };

  const getSortArrow = (dataset, metric) => {
    if (sortConfig?.key === `${dataset}-${metric}`) {
      return sortConfig.direction === 'asc' ? '↑' : '↓';
    }
    return '↕';
  };

  return (
    <div className='Leaderboard'>
      <h1 className='leaderboard-title'>Leaderboard</h1>

      <div className='table-container'>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              {datasets.map((dataset) => (
                <th key={dataset} colSpan={metrics[dataset]?.length || 0}>
                  <a href={datasetUrl[dataset]?.url}>{dataset}</a>
                </th>
              ))}
            </tr>
            <tr>
              <th></th>
              {datasets.map((dataset) =>
                metrics[dataset]?.map((metric) => (
                  <th
                    key={`${dataset}-${metric}`}
                    onClick={() => handleSort(dataset, metric)}
                  >
                    {metric} {getSortArrow(dataset, metric)}
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                <td>
                  <a href={item.url}>{item.model}</a>
                </td>
                {datasets.map((dataset) =>
                  metrics[dataset]?.map((metric) => (
                    <td
                      key={`${dataset}-${metric}`}
                      className={
                        defaultMetrics[dataset] === metric
                          ? 'highlight-column'
                          : ''
                      }
                    >
                      {item.scores[dataset]?.[metric] || '-'}
                    </td>
                  )),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
export default Leaderboard;
