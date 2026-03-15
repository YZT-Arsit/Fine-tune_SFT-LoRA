# Books to Scrape Data Pipeline

一个合规、可复现的电商数据采集与解析项目，针对公开练习站点 `https://books.toscrape.com` 生成：

- 原始 HTML（可选保存）
- 结构化商品 JSONL
- 用于 SFT/LoRA 的 instruction/input/output JSONL

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置 API（推荐 `.env`）

在项目根目录创建 `.env`：

```bash
cat > .env <<'EOF'
LLM_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=YOUR_KEY_HERE
LLM_MODEL=qwen-plus
LLM_GUIDED_JSON=auto
EOF
```

说明：

- 运行 `python -m src.main ...` 时会自动加载项目根目录的 `.env`
- 已使用 `override=False`，所以如果你在 shell 里先 `export` 了同名变量，shell 变量优先
- `.env` 已加入 `.gitignore`，不会被默认提交

## 抓取

```bash
python -m src.main crawl --max_pages 50 --out outputs
```

说明：

- 默认严格检查 `robots.txt`
- 默认启用限速（1~2 秒随机抖动）
- 失败自动重试并指数退避
- 支持断点续爬；重复运行不会重复写入相同 `product_url`
- 如需保存详情页 HTML：

```bash
python -m src.main crawl --max_pages 50 --out outputs --save-html
```

## 构建 SFT 数据

```bash
python -m src.main build_sft --in outputs/parsed/products.jsonl --out outputs/sft/train.jsonl
```

输出格式为 JSONL，每行包含：

- `instruction`
- `input`
- `output`（JSON 字符串）

## 统计

```bash
python -m src.main stats --in outputs/parsed/products.jsonl
```

统计内容：

- 样本数
- 字段缺失率
- `description` 平均长度
- `category` 分布
- `price` 简单分布

## Baseline 评测

```bash
python -m src.main baseline \
  --val outputs/sft/val.jsonl \
  --pred_out outputs/baseline/baseline_predictions.jsonl \
  --report_out outputs/baseline/baseline_eval_report.json \
  --mode api \
  --model_name qwen-plus
```

推荐直接依赖 `.env`，无需手动 `export`。

默认推荐 `api` 模式，适合 Mac Air。

环境变量：

```bash
export LLM_API_BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="your_api_key"
export LLM_MODEL="qwen-plus"
export LLM_GUIDED_JSON="auto"
```

也可以使用自建 vLLM：

```bash
export LLM_API_BASE="http://127.0.0.1:8000/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_GUIDED_JSON="auto"
```

说明：

- `api` 模式固定使用 OpenAI 兼容的 `POST /chat/completions`
- `local` 模式是可选能力，需要你自行安装 `transformers`
- 推理时不会把 ground truth 提供给模型，只发送 `instruction + input`
- 预测结果写入 `baseline_predictions.jsonl`
- 汇总评测写入 `baseline_eval_report.json`
- 当端点看起来像 vLLM（如 `localhost/127.0.0.1`）且 `LLM_GUIDED_JSON=auto` 时，会自动尝试启用 `guided_json`
- `guided_json` 仅适用于支持该能力的 vLLM 兼容端点；若服务端不支持，会自动降级回纯 prompt + 解析兜底

## 输出目录

运行后默认输出到：

- `outputs/raw_html/`
- `outputs/parsed/products.jsonl`
- `outputs/sft/train.jsonl`
- `outputs/logs/`

## 参数说明

### `crawl`

- `--max_pages`: 最多抓取多少个列表分页
- `--out`: 输出目录根路径
- `--save-html`: 是否保存详情页原始 HTML
- `--user-agent`: 自定义 User-Agent

### `build_sft`

- `--in`: 输入的结构化商品 JSONL
- `--out`: 输出的 SFT JSONL

### `baseline`

- `--val`: 验证集 JSONL
- `--pred_out`: baseline 预测输出 JSONL
- `--report_out`: baseline 评测报告 JSON
- `--mode`: `api` 或 `local`
- `--model_name`: 模型名（可覆盖 `LLM_MODEL`）
- `--max_samples`: 调试时限制样本数
- `--max_tokens`: 生成 token 上限，默认 `512`

## Badcase 分析

对验证集 gold 与 baseline 预测进行字段级差异分析，并输出 badcase 汇总：

```bash
python -m src.analyze_badcases \
  --gold outputs/sft/val.jsonl \
  --pred outputs/baseline/baseline_qwen7b_predictions.jsonl \
  --summary_out outputs/analysis/badcase_summary.json \
  --badcases_out outputs/analysis/badcases.jsonl
```

输出：

- `outputs/analysis/badcase_summary.json`
- `outputs/analysis/badcases.jsonl`

## 训练集增强

基于 badcase 分析结果，对 `train.jsonl` 做回环增强（字段扰动、区块顺序扰动、缺失字段鲁棒性增强等）：

```bash
python -m src.augment_sft_data \
  --train_in outputs/sft/train.jsonl \
  --badcase_in outputs/analysis/badcases.jsonl \
  --train_out outputs/sft/train_augmented.jsonl \
  --max_aug_per_sample 2 \
  --dedup \
  --seed 42
```

输出：

- `outputs/sft/train_augmented.jsonl`
- `outputs/sft/train_augmented_report.json`

## LoRA 微调训练

LoRA/QLoRA 训练脚本位于 `src/train_lora.py`，目标模型默认 `Qwen/Qwen2.5-7B-Instruct`。

建议先安装训练依赖：

```bash
pip install -r requirements-lora.txt
```

默认会优先走 QLoRA（适配单卡 24GB 显存），并在训练目录中保存：

- `final_adapter/`（adapter + tokenizer）
- `training_config.json`
- `train_metrics.json`
- `eval_metrics.json`

如果启用 `--do_generation_eval`，会额外输出：

- `outputs/lora_eval/val_predictions.jsonl`
- `outputs/lora_eval/val_generation_report.json`

1) QLoRA 训练（推荐）：

```bash
python -m src.train_lora \
  --train_file outputs/sft/train.jsonl \
  --val_file outputs/sft/val.jsonl \
  --output_dir checkpoints/lora_qwen2.5_7b \
  --mode qlora
```

2) 训练完成后带生成评测：

```bash
python -m src.train_lora \
  --train_file outputs/sft/train.jsonl \
  --val_file outputs/sft/val.jsonl \
  --output_dir checkpoints/lora_qwen2.5_7b \
  --mode qlora \
  --do_generation_eval \
  --max_eval_samples 50
```

### 合并 LoRA Adapter

将 `final_adapter` 合并回 base model，得到可直接部署的完整模型权重：

```bash
python -m src.merge_lora \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir checkpoints/lora_qwen2.5_7b/final_adapter \
  --output_dir checkpoints/lora_qwen2.5_7b/merged_model \
  --dtype auto
```

也可以直接把训练输出目录传给 `--adapter_dir`，脚本会优先自动查找：

- `final_adapter/`
- 最新的 `checkpoint-*`

### `stats`

- `--in`: 输入的结构化商品 JSONL

## 合规说明

- 仅访问公开页面
- 不登录、不绕过验证码、不使用代理
- 先检查 `robots.txt`，若禁止则跳过
- 默认低速串行抓取，降低站点压力
