# Baseline vs LoRA 对比

- Baseline: `/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct`
- LoRA: `final_adapter`
- Baseline evaluated: `118`
- LoRA evaluated: `118`

## Headline Metrics

- `json_parse_rate`: baseline=1.0 | lora=1.0 | delta=0.0 | improved=True
- `schema_valid_rate`: baseline=1.0 | lora=1.0 | delta=0.0 | improved=True

## Interview Summary

我先用基座模型建立了结构化抽取 baseline，再基于 badcase 分析做数据治理与回环增强，最后用 QLoRA 在单卡 24GB 环境完成微调，并用统一验证集做同口径评测。

LoRA 相比 baseline 的 JSON 解析率变化 +0.0000，Schema 合法率变化 +0.0000。

### Top Field Gains
- `rating_match_rate`: baseline=0.0 | lora=0.1949 | delta=0.1949
- `category_exact_match_rate`: baseline=0.9237 | lora=1.0 | delta=0.0763
- `availability_text_match_rate`: baseline=1.0 | lora=1.0 | delta=0.0
- `currency_match_rate`: baseline=1.0 | lora=1.0 | delta=0.0
- `in_stock_match_rate`: baseline=1.0 | lora=1.0 | delta=0.0

### Top Numeric Improvements
- `price_abs_error_avg`: baseline=0.0 | lora=0.0 | delta=0.0
- `price_abs_error_p50`: baseline=0.0 | lora=0.0 | delta=0.0
- `price_abs_error_p95`: baseline=0.0 | lora=0.0 | delta=0.0
- `price_excl_tax_abs_error_avg`: baseline=0.0 | lora=0.0 | delta=0.0
- `price_excl_tax_abs_error_p50`: baseline=0.0 | lora=0.0 | delta=0.0
