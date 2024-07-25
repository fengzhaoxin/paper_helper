# 模型量化
首先进入ryzenai-transformers环境，并设置环境变量

### 量化
python run_awq.py --w_bit 3 --lm_head --flash_attention --model ../Agent-FLAN-7b --output ./Agent-FLAN-7b

### 测试
python run_chat.py --model Agent-FLAN-7b_w_3_fa_lm --tokenizer ../Agent-FLAN-7b
