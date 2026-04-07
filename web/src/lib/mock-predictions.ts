/** Mock data for the Prediction Browser. */

export interface ModelPrediction {
  modelId: string;
  modelName: string;
  prediction: string;
  score: number; // 1 = correct, 0 = incorrect
}

export interface Sample {
  id: string;
  /** Placeholder image URL or null for a colored placeholder */
  imageUrl: string | null;
  /** Hex color for the placeholder when imageUrl is null */
  placeholderColor: string;
  question: string;
  groundTruth: string;
  predictions: ModelPrediction[];
}

export interface Task {
  id: string;
  name: string;
  sampleCount: number;
}

export const TASKS: Task[] = [
  { id: "jmmmu", name: "JMMMU", sampleCount: 150 },
  { id: "japanese-heron-bench", name: "Japanese Heron Bench", sampleCount: 85 },
  { id: "jdocqa", name: "JDocQA", sampleCount: 200 },
  { id: "mmmu", name: "MMMU", sampleCount: 300 },
];

export const MODELS = [
  { id: "qwen2.5-vl-72b", name: "Qwen2.5-VL-72B" },
  { id: "internvl3-78b", name: "InternVL3-78B" },
  { id: "gpt-4o", name: "GPT-4o" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash" },
  { id: "llava-next-72b", name: "LLaVA-NeXT-72B" },
];

export const MOCK_SAMPLES: Sample[] = [
  {
    id: "jmmmu-001",
    imageUrl: null,
    placeholderColor: "#d4c5a9",
    question:
      "この回路図において、抵抗R1とR2が直列に接続されている場合、合成抵抗はいくらになりますか？R1=10\u03A9, R2=20\u03A9とします。",
    groundTruth: "30\u03A9",
    predictions: [
      { modelId: "qwen2.5-vl-72b", modelName: "Qwen2.5-VL-72B", prediction: "30\u03A9", score: 1 },
      { modelId: "internvl3-78b", modelName: "InternVL3-78B", prediction: "30\u03A9", score: 1 },
      { modelId: "gpt-4o", modelName: "GPT-4o", prediction: "30\u03A9", score: 1 },
      { modelId: "gemini-2.0-flash", modelName: "Gemini 2.0 Flash", prediction: "200\u03A9", score: 0 },
      { modelId: "llava-next-72b", modelName: "LLaVA-NeXT-72B", prediction: "30\u03A9", score: 1 },
    ],
  },
  {
    id: "jmmmu-002",
    imageUrl: null,
    placeholderColor: "#b8c5d4",
    question:
      "この地図に示されている都市の名前は何ですか？また、その都市が属する都道府県名も答えてください。",
    groundTruth: "京都市、京都府",
    predictions: [
      { modelId: "qwen2.5-vl-72b", modelName: "Qwen2.5-VL-72B", prediction: "京都市、京都府", score: 1 },
      { modelId: "internvl3-78b", modelName: "InternVL3-78B", prediction: "大阪市、大阪府", score: 0 },
      { modelId: "gpt-4o", modelName: "GPT-4o", prediction: "京都市、京都府", score: 1 },
      { modelId: "gemini-2.0-flash", modelName: "Gemini 2.0 Flash", prediction: "京都市、京都府", score: 1 },
      { modelId: "llava-next-72b", modelName: "LLaVA-NeXT-72B", prediction: "奈良市、奈良県", score: 0 },
    ],
  },
  {
    id: "jmmmu-003",
    imageUrl: null,
    placeholderColor: "#c5d4b8",
    question:
      "このグラフから読み取れる2023年の売上高はいくらですか？単位は億円で答えてください。",
    groundTruth: "42億円",
    predictions: [
      { modelId: "qwen2.5-vl-72b", modelName: "Qwen2.5-VL-72B", prediction: "42億円", score: 1 },
      { modelId: "internvl3-78b", modelName: "InternVL3-78B", prediction: "45億円", score: 0 },
      { modelId: "gpt-4o", modelName: "GPT-4o", prediction: "42億円", score: 1 },
      { modelId: "gemini-2.0-flash", modelName: "Gemini 2.0 Flash", prediction: "42億円", score: 1 },
      { modelId: "llava-next-72b", modelName: "LLaVA-NeXT-72B", prediction: "40億円", score: 0 },
    ],
  },
  {
    id: "jmmmu-004",
    imageUrl: null,
    placeholderColor: "#d4b8c5",
    question:
      "この化学構造式が表す化合物の名称をIUPAC命名法で答えてください。",
    groundTruth: "2-メチルプロパン-1-オール",
    predictions: [
      { modelId: "qwen2.5-vl-72b", modelName: "Qwen2.5-VL-72B", prediction: "2-メチルプロパン-1-オール", score: 1 },
      { modelId: "internvl3-78b", modelName: "InternVL3-78B", prediction: "イソブタノール", score: 0 },
      { modelId: "gpt-4o", modelName: "GPT-4o", prediction: "2-メチルプロパン-1-オール", score: 1 },
      { modelId: "gemini-2.0-flash", modelName: "Gemini 2.0 Flash", prediction: "2-methylpropan-1-ol", score: 0 },
      { modelId: "llava-next-72b", modelName: "LLaVA-NeXT-72B", prediction: "2-メチルプロパン-1-オール", score: 1 },
    ],
  },
  {
    id: "jmmmu-005",
    imageUrl: null,
    placeholderColor: "#c5b8d4",
    question:
      "この絵画の作者と作品名を答えてください。日本語で回答すること。",
    groundTruth: "葛飾北斎「富嶽三十六景 神奈川沖浪裏」",
    predictions: [
      { modelId: "qwen2.5-vl-72b", modelName: "Qwen2.5-VL-72B", prediction: "葛飾北斎「富嶽三十六景 神奈川沖浪裏」", score: 1 },
      { modelId: "internvl3-78b", modelName: "InternVL3-78B", prediction: "葛飾北斎「富嶽三十六景 神奈川沖浪裏」", score: 1 },
      { modelId: "gpt-4o", modelName: "GPT-4o", prediction: "葛飾北斎「富嶽三十六景 神奈川沖浪裏」", score: 1 },
      { modelId: "gemini-2.0-flash", modelName: "Gemini 2.0 Flash", prediction: "葛飾北斎「神奈川沖浪裏」", score: 0 },
      { modelId: "llava-next-72b", modelName: "LLaVA-NeXT-72B", prediction: "歌川広重「東海道五十三次」", score: 0 },
    ],
  },
];
