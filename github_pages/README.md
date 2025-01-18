# llm-jp-eval-mm.github.io

This repository is a source code for the llm-jp-eval-mm leaderboard website.
[llm-jp-eval-mm](https://github.com/llm-jp/llm-jp-eval-mm) is used to evaluate the VLMs on the Japanese benchmark.

## How to develop
```bash
cd github_pages
sudo apt install -y nodejs npm
sudo npm install n -g
npm install
npm run start
```

You may need to remove "homepage" from `github_pages/package.json` to start in the local environment.
```diff
{
  "name": "github_pages",
  "version": "0.1.0",
--  "homepage": "https://llm-jp.github.io/llm-jp-eval-mm",
}
```

## How to deploy
```bash
cd github_pages
npm run deploy
```

## Add benchmark results to the leaderboard
Please add the benchmark results to the `github_pages/public/leaderboard.json` file.
The format of the benchmark results is as follows.
```json
  {
    "model": "Japanese InstructBLIP Alpha",
    "url": "https://huggingface.co/stabilityai/japanese-instructblip-alpha",
    "scores": {
      "Heron": {
        "conv": 22.8,
        "detail": 24.1,
        "complex": 19.5,
        "overall": 22.7
      },
      "JVB-ItW": { "llm": 1.31, "rouge": 13.8 },
      "MulIm-VQA": { "llm": 2.5, "rouge": 25.0 },
      "JDocQA": { "Acc": 0.123, "llm": 1.9 },
      "JMMMU": { "Acc": 0.271 }
    }
  },
```

## Format the code
```bash
npx prettier --write "./**/*.{js,jsx,ts,tsx,css,html}"
```


## Reference
This repository refers to the following repositories. Thank you for your great work.
- https://github.com/MMMU-Japanese-Benchmark/JMMMU
- https://github.com/MMMU-Benchmark/mmmu-benchmark.github.io