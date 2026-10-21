"""
check_chat_template.py
======================
对比 StarVLA .pt checkpoint（VLM部分）和原生 Qwen3.5 的 chat_template，
找出训练/推理不一致的根本原因。

用法:
    python check_chat_template.py \
        --ckpt   playground/Checkpoints/.../steps_25000_pytorch_model.pt \
        --base_vlm  playground/Pretrained_models/Qwen3.5-4B \
        --vlm_prefix qwen_vl_interface
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

SEP = "═" * 70

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_special_tokens(tokenizer, label: str):
    """打印关键 special token 的 id，重点检查 im_end"""
    print(f"\n── Special Tokens [{label}] ──")
    tokens_to_check = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
    ]
    for tok in tokens_to_check:
        tid = tokenizer.convert_tokens_to_ids(tok)
        is_eos = (tid == tokenizer.eos_token_id)
        flag = "  ← eos_token ✅" if is_eos else ("  ← ⚠️  NOT eos_token" if tok == "<|im_end|>" else "")
        print(f"  {tok:<25} id={tid}{flag}")

    print(f"\n  eos_token       = '{tokenizer.eos_token}'  (id={tokenizer.eos_token_id})")
    print(f"  pad_token       = '{tokenizer.pad_token}'  (id={tokenizer.pad_token_id})")
    print(f"  model_max_length= {tokenizer.model_max_length}")


def print_template(tokenizer, label: str):
    """打印完整 chat_template 并高亮关键部分"""
    print(f"\n── chat_template [{label}] ──")
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl is None:
        print("  ⚠️  chat_template = None（未设置）")
        return

    print(f"  长度: {len(tmpl)} 字符")
    print()

    # 逐行缩进打印
    for line in tmpl.replace("}{", "}\n{").splitlines():
        print(f"  {line}")

    # 关键词检查
    print()
    keywords = {
        "<think>":      "Qwen3 thinking模式标志（训练时若无此token，推理会乱）",
        "<|im_end|>":   "对话停止符（训练和推理必须一致）",
        "add_generation_prompt": "推理时拼接 <|im_start|>assistant\\n 的开关",
        "system":       "system role 支持",
    }
    for kw, desc in keywords.items():
        found = kw in tmpl
        status = "✅ 存在" if found else "❌ 不存在"
        print(f"  [{status}] '{kw}' — {desc}")


def render_sample_prompt(tokenizer, label: str):
    """用一条最简单的对话渲染出实际文本，直观对比"""
    print(f"\n── 渲染示例（单轮对话）[{label}] ──")
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 1+1?"},
        {"role": "assistant", "content": "2"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(repr(rendered))
    except Exception as e:
        print(f"  ⚠️  渲染失败: {e}")

    # 再渲染一次 add_generation_prompt=True，看推理时末尾是什么
    print(f"\n── 渲染示例（推理时末尾，add_generation_prompt=True）[{label}] ──")
    messages_infer = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is 1+1?"},
    ]
    try:
        rendered_infer = tokenizer.apply_chat_template(
            messages_infer,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(repr(rendered_infer[-120:]))   # 只打印末尾，看触发词
    except Exception as e:
        print(f"  ⚠️  渲染失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 从 .pt 提取 tokenizer config（不加载整个模型）
# ─────────────────────────────────────────────────────────────────────────────

def extract_vlm_prefix(ckpt_path: str, vlm_prefix: str) -> dict:
    """只看 key 前缀分布，不真正加载权重张量"""
    print(f"\n  Loading checkpoint (weights_only=False for inspection)...")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    dist = Counter(k.split(".")[0] for k in ckpt)
    print(f"  Checkpoint top-level prefixes:")
    for p, c in dist.most_common(10):
        ex = next(k for k in ckpt if k.split(".")[0] == p)
        print(f"    {p:<50} {c:>6}  e.g. {ex}")

    # 检查 checkpoint 里是否保存了 tokenizer 相关配置
    print(f"\n  检查 checkpoint 内是否有 chat_template / tokenizer 字段:")
    for key in ckpt:
        if any(kw in key.lower() for kw in ["chat_template", "tokenizer", "template"]):
            print(f"    ⚠️  Found: {key}")

    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# 对比两个 tokenizer 的 template 差异
# ─────────────────────────────────────────────────────────────────────────────

def compare_templates(tok_base, tok_ckpt_base):
    """
    .pt checkpoint 本身不存 tokenizer，
    所以 VLA 的 tokenizer 实际上就是 base_vlm 的 tokenizer。
    这里对比：
      1. 原生 tokenizer（推理时默认用的）
      2. 训练时被覆盖的 template（vlm_dataset.py 里硬编码的）
    """
    section("对比：原生 template vs 训练时使用的 template")

    training_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    import copy
    tok_train = copy.deepcopy(tok_base)
    tok_train.chat_template = training_template

    print("\n[原生 template 渲染结果]")
    render_sample_prompt(tok_base, "原生 Qwen3.5")

    print("\n[训练时 template 渲染结果]")
    render_sample_prompt(tok_train, "训练时覆盖的 template")

    # 检查关键差异
    native_tmpl  = tok_base.chat_template or ""
    print("\n── 关键差异分析 ──")

    checks = [
        ("<think>",    native_tmpl, training_template,
         "Qwen3 thinking token：原生有但训练时无 → 推理时模型不认识格式"),
        ("<|im_end|>", native_tmpl, training_template,
         "停止符：两边都有 ✅"),
        ("add_generation_prompt", native_tmpl, training_template,
         "推理时 prompt 末尾触发词"),
    ]

    all_ok = True
    for kw, t1, t2, desc in checks:
        in_native   = kw in t1
        in_training = kw in t2
        if in_native != in_training:
            print(f"  ⚠️  MISMATCH '{kw}': 原生={in_native}, 训练时={in_training}  → {desc}")
            all_ok = False
        else:
            print(f"  ✅  MATCH    '{kw}': 两边均={'存在' if in_native else '不存在'}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,  help=".pt checkpoint 路径")
    parser.add_argument("--base_vlm",    required=True,  help="原生 Qwen3.5 路径")
    parser.add_argument("--vlm_prefix",  default="qwen_vl_interface",
                        help="checkpoint 里 VLM 部分的 key 前缀")
    args = parser.parse_args()

    # ── Step 1: 检查 .pt checkpoint 结构 ────────────────────────────────
    section("Step 1: 检查 .pt Checkpoint 结构")
    if not os.path.exists(args.ckpt):
        print(f"  ❌ ckpt 路径不存在: {args.ckpt}")
        return
    ckpt = extract_vlm_prefix(args.ckpt, args.vlm_prefix)

    # ── Step 2: 加载原生 tokenizer（不加载模型权重）────────────────────
    section("Step 2: 加载原生 Qwen3.5 Tokenizer")
    print(f"  路径: {args.base_vlm}")
    try:
        tok_native = AutoTokenizer.from_pretrained(
            args.base_vlm,
            trust_remote_code=True,
            use_fast=False,
        )
        print(f"  ✅ 加载成功: {type(tok_native).__name__}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return

    # ── Step 3: 打印原生 tokenizer 的 template ──────────────────────────
    section("Step 3: 原生 Qwen3.5 chat_template")
    print_template(tok_native, "原生 Qwen3.5")
    check_special_tokens(tok_native, "原生 Qwen3.5")
    render_sample_prompt(tok_native, "原生 Qwen3.5")

    # ── Step 4: 打印训练时覆盖的 template ───────────────────────────────
    section("Step 4: 训练时（vlm_dataset.py）使用的 chat_template")
    training_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    import copy
    tok_train = copy.deepcopy(tok_native)
    tok_train.chat_template = training_template
    print_template(tok_train, "训练时覆盖的 template")
    check_special_tokens(tok_train, "训练时覆盖的 template（special tokens 不变）")
    render_sample_prompt(tok_train, "训练时覆盖的 template")

    # ── Step 5: 对比差异 ────────────────────────────────────────────────
    all_ok = compare_templates(tok_native, tok_train)

    # ── Step 6: 给出修复建议 ────────────────────────────────────────────
    section("Step 6: 修复建议")

    im_end_id = tok_native.convert_tokens_to_ids("<|im_end|>")
    eos_id    = tok_native.eos_token_id

    if im_end_id != eos_id:
        print(f"""
  ❌ <|im_end|> (id={im_end_id}) ≠ eos_token (id={eos_id})
  → 推理时 generate() 必须显式指定：

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=[{eos_id}, {im_end_id}],   # ← 两个都要加
        repetition_penalty=1.05,
    )
""")
    else:
        print(f"  ✅ <|im_end|> == eos_token (id={im_end_id})，generate 可自动停止")

    if not all_ok:
        print(f"""
  ❌ chat_template 训练/推理不一致！
  → 在推理脚本的 __init__ 里加：

    self.processor.tokenizer.chat_template = (
        "{{% for message in messages %}}"
        "{{{{'<|im_start|>' + message['role'] + '\\\\n' + message['content'] + '<|im_end|>' + '\\\\n'}}}}"
        "{{% endfor %}}"
        "{{% if add_generation_prompt %}}{{{{ '<|im_start|>assistant\\\\n' }}}}{{% endif %}}"
    )
""")
    else:
        print("  ✅ chat_template 训练/推理一致")

    print(f"\n{'═'*70}")
    print("  诊断完成")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()