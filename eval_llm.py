import time
import argparse
import json
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
import torch
warnings.filterwarnings('ignore')


LIVECODEBENCH_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
LIVECODEBENCH_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
LIVECODEBENCH_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def logit_lens_explain(model, tokenizer, input_ids, attention_mask, top_k=5):
    """Logit lens: print each layer's top-k next-token predictions for the last input position."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, logit_lens=True)
    layer_logits = outputs.layer_logits
    last_pos = input_ids.shape[1] - 1
    print(f'[Logit Lens] 输入 {input_ids.shape[1]} tokens，逐层展示位置 {last_pos} 的下一个token预测（Top-{top_k}）:\n')
    for i, lg in enumerate(layer_logits):
        probs = torch.softmax(lg.float(), dim=-1)[0, last_pos]
        top_probs, top_ids = torch.topk(probs, top_k)
        preds = ' | '.join(
            f'{tokenizer.decode([tid], skip_special_tokens=True)!r} {p * 100:.1f}%'
            for tid, p in zip(top_ids.tolist(), top_probs.tolist())
        )
        final = ' (final)' if i == len(layer_logits) - 1 else ''
        print(f'  Layer {i:>2}{final}: {preds}')
    return outputs

def init_model(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from model.model_lora import apply_lora, load_lora
    from trainer.trainer_utils import get_model_params

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        if args.config_path:
            with open(args.config_path, 'r', encoding='utf-8') as config_file:
                config_kwargs = json.load(config_file)
        else:
            config_kwargs = {
                'hidden_size': args.hidden_size,
                'num_hidden_layers': args.num_hidden_layers,
                'use_moe': bool(args.use_moe),
                'inference_rope_scaling': args.inference_rope_scaling,
                'residual_type': args.residual_type,
                'hc_mult': args.hc_mult,
                'hc_sinkhorn_iters': args.hc_sinkhorn_iters,
                'attnres_variant': args.attnres_variant,
                'attnres_block_size': args.attnres_block_size,
            }
        architecture = config_kwargs.get('model_architecture', args.model_architecture)
        if architecture == 'linear':
            from model.model_instinct_linear import (
                InstinctConfig as LinearInstinctConfig,
                InstinctForCausalLM as LinearInstinctForCausalLM,
            )
            model = LinearInstinctForCausalLM(LinearInstinctConfig(**config_kwargs))
        elif architecture == 'looped':
            from model.model_instinct_loop import (
                InstinctConfig as LoopedInstinctConfig,
                InstinctForCausalLM as LoopedInstinctForCausalLM,
            )
            model = LoopedInstinctForCausalLM(LoopedInstinctConfig(**config_kwargs))
        else:
            model = InstinctForCausalLM(InstinctConfig(**config_kwargs))
        moe_suffix = '_moe' if model.config.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{model.config.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{model.config.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def format_livecodebench_prompt(problem):
    """Build the generic code-generation prompt used by LiveCodeBench."""
    prompt = f"### Question:\n{problem['question_content']}\n\n"
    starter_code = problem.get('starter_code') or ''
    if starter_code:
        prompt += f"### Format: {LIVECODEBENCH_FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LIVECODEBENCH_FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_livecodebench_code(model_output):
    """Match LiveCodeBench's generic-instruct extraction: use the last fenced block."""
    output_lines = model_output.splitlines()
    fence_lines = [i for i, line in enumerate(output_lines) if '```' in line]
    if len(fence_lines) < 2:
        return ''
    return '\n'.join(output_lines[fence_lines[-2] + 1:fence_lines[-1]]).strip()


def _parse_iso_date(value, flag_name):
    if value is None:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError as exc:
        raise ValueError(f'{flag_name} 必须使用 YYYY-MM-DD 格式，收到: {value}') from exc


def _read_local_livecodebench_dataset(path):
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f'LiveCodeBench 数据文件不存在: {dataset_path}')
    with dataset_path.open('r', encoding='utf-8') as dataset_file:
        if dataset_path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in dataset_file if line.strip()]
        payload = json.load(dataset_file)
    if not isinstance(payload, list):
        raise ValueError('--lcb_dataset_path 指向的 JSON 顶层必须是数组')
    return payload


def load_livecodebench_problems(args):
    """Load, filter and sort LiveCodeBench code-generation problems."""
    if args.lcb_dataset_path:
        problems = _read_local_livecodebench_dataset(args.lcb_dataset_path)
    else:
        try:
            dataset = load_dataset(
                'livecodebench/code_generation_lite',
                name=args.lcb_release_version,
                split='test',
                trust_remote_code=True,
            )
        except Exception as exc:
            message = str(exc)
            if 'Dataset scripts are no longer supported' in message:
                raise RuntimeError(
                    '当前 datasets 版本已移除数据集脚本支持。请按 requirements.txt '
                    '安装 datasets==3.6.0，或用 --lcb_dataset_path 指定已下载的 JSON/JSONL。'
                ) from exc
            raise RuntimeError(
                '无法加载 LiveCodeBench 数据集。可检查 Hugging Face 网络/镜像设置，'
                '或用 --lcb_dataset_path 指定本地 JSON/JSONL。'
            ) from exc
        problems = [dict(problem) for problem in dataset]

    required_fields = {'question_id', 'question_content'}
    for index, problem in enumerate(problems):
        missing = required_fields.difference(problem)
        if missing:
            raise ValueError(f'LiveCodeBench 第 {index} 条记录缺少字段: {sorted(missing)}')

    start_date = _parse_iso_date(args.lcb_start_date, '--lcb_start_date')
    end_date = _parse_iso_date(args.lcb_end_date, '--lcb_end_date')
    if start_date and end_date and start_date > end_date:
        raise ValueError('--lcb_start_date 不能晚于 --lcb_end_date')

    if start_date or end_date:
        filtered = []
        for problem in problems:
            if not problem.get('contest_date'):
                raise ValueError('使用日期过滤时，每条记录都必须包含 contest_date')
            contest_date = datetime.fromisoformat(problem['contest_date'])
            if start_date and contest_date < start_date:
                continue
            if end_date and contest_date > end_date:
                continue
            filtered.append(problem)
        problems = filtered

    problems.sort(key=lambda problem: str(problem['question_id']))
    if args.lcb_limit:
        problems = problems[:args.lcb_limit]
    return problems


def _write_json_atomic(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write('\n')
    os.replace(temporary_path, output_path)


def _load_livecodebench_resume(path, selected_ids, num_samples):
    output_path = Path(path)
    if not output_path.exists():
        return {}
    with output_path.open('r', encoding='utf-8') as output_file:
        saved_rows = json.load(output_file)
    if not isinstance(saved_rows, list):
        raise ValueError(f'已有输出不是 JSON 数组: {output_path}')

    saved = {}
    for row in saved_rows:
        if not isinstance(row, dict) or 'question_id' not in row or 'code_list' not in row:
            raise ValueError(f'已有输出不符合 LiveCodeBench custom evaluator 格式: {output_path}')
        question_id = str(row['question_id'])
        if question_id in saved:
            raise ValueError(f'已有输出包含重复 question_id: {question_id}')
        if not isinstance(row['code_list'], list):
            raise ValueError(f'{question_id} 的 code_list 必须是数组')
        if len(row['code_list']) > num_samples:
            raise ValueError(
                f'{question_id} 已有 {len(row["code_list"])} 个样本，超过本次要求的 '
                f'{num_samples}；请改用新的 --lcb_output 路径'
            )
        saved[question_id] = list(row['code_list'])

    unexpected = set(saved).difference(selected_ids)
    if unexpected:
        raise ValueError(
            '已有输出与本次数据范围不一致（存在额外 question_id）；'
            '请改用新的 --lcb_output 路径'
        )
    return saved


def _livecodebench_input_text(args, tokenizer, problem):
    user_prompt = format_livecodebench_prompt(problem)
    prompt_style = args.lcb_prompt_style
    if prompt_style == 'auto':
        prompt_style = 'base' if args.load_from == 'model' and 'pretrain' in args.weight else 'chat'
    if prompt_style == 'base':
        return f'{LIVECODEBENCH_SYSTEM_PROMPT}\n\n{user_prompt}'
    messages = [
        {'role': 'system', 'content': LIVECODEBENCH_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(args.open_thinking),
    )


def _generation_kwargs(args, tokenizer, num_return_sequences=1):
    do_sample = args.temperature > 0
    if num_return_sequences > 1 and not do_sample:
        raise ValueError('生成多个样本时 --temperature 必须大于 0')
    kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': do_sample,
        'num_return_sequences': num_return_sequences,
        'repetition_penalty': 1,
    }
    if tokenizer.pad_token_id is not None:
        kwargs['pad_token_id'] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs['eos_token_id'] = tokenizer.eos_token_id
    if do_sample:
        kwargs['top_p'] = args.top_p
        kwargs['temperature'] = args.temperature
    if args.early_exit:
        kwargs['early_exit'] = True
        kwargs['exit_threshold'] = args.exit_threshold
    return kwargs


def run_livecodebench_generation(args, model, tokenizer):
    if args.lcb_num_samples > 1 and args.temperature <= 0:
        raise ValueError('每题生成多个样本时 --temperature 必须大于 0')
    problems = load_livecodebench_problems(args)
    if not problems:
        raise ValueError('筛选后没有 LiveCodeBench 题目')

    selected_ids = {str(problem['question_id']) for problem in problems}
    saved = (
        _load_livecodebench_resume(args.lcb_output, selected_ids, args.lcb_num_samples)
        if args.lcb_resume else {}
    )
    rows = []
    total_generated_tokens = 0
    started_at = time.time()

    print(
        f'[LiveCodeBench] {len(problems)} problems, '
        f'{args.lcb_num_samples} sample(s) each, release={args.lcb_release_version}'
    )
    for problem_index, problem in enumerate(problems):
        question_id = str(problem['question_id'])
        code_list = saved.get(question_id, [])
        if len(code_list) < args.lcb_num_samples:
            input_text = _livecodebench_input_text(args, tokenizer, problem)
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True).to(args.device)
            input_length = inputs['input_ids'].shape[1]

            for sample_index in range(len(code_list), args.lcb_num_samples):
                setup_seed(args.lcb_seed + problem_index * args.lcb_num_samples + sample_index)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        inputs=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        **_generation_kwargs(args, tokenizer),
                    )
                response_ids = generated_ids[0][input_length:]
                total_generated_tokens += len(response_ids)
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                code = extract_livecodebench_code(response)
                code_list.append(code)
                if not code:
                    print(f'[LiveCodeBench] warning: {question_id} sample {sample_index + 1} 未提取到代码块')

        saved[question_id] = code_list
        rows = [
            {'question_id': str(item['question_id']), 'code_list': saved.get(str(item['question_id']), [])}
            for item in problems
            if str(item['question_id']) in saved
        ]
        _write_json_atomic(args.lcb_output, rows)
        print(f'[LiveCodeBench] {problem_index + 1}/{len(problems)} {question_id}')

    elapsed = max(time.time() - started_at, 1e-9)
    print(f'[LiveCodeBench] generations saved to {Path(args.lcb_output).resolve()}')
    if args.show_speed:
        print(f'[LiveCodeBench] {total_generated_tokens / elapsed:.2f} generated tokens/s')
    return rows


def build_livecodebench_evaluator_command(args):
    command = [
        sys.executable,
        '-m',
        'lcb_runner.runner.custom_evaluator',
        '--custom_output_file',
        str(Path(args.lcb_output).resolve()),
        '--scenario',
        'codegeneration',
        '--release_version',
        args.lcb_release_version,
        '--num_process_evaluate',
        str(args.lcb_num_process_evaluate),
        '--timeout',
        str(args.lcb_timeout),
    ]
    if args.lcb_start_date:
        command.extend(['--start_date', args.lcb_start_date])
    if args.lcb_end_date:
        command.extend(['--end_date', args.lcb_end_date])
    return command


def run_livecodebench_evaluator(args):
    if args.lcb_limit:
        raise ValueError('官方 custom evaluator 要求完整数据范围，--lcb_evaluate 不能与 --lcb_limit 同用')
    output_path = Path(args.lcb_output)
    if not output_path.is_file():
        raise FileNotFoundError(f'LiveCodeBench 生成文件不存在: {output_path}')
    runner_path = Path(args.lcb_runner_path).resolve() if args.lcb_runner_path else None
    if runner_path and not (runner_path / 'lcb_runner').is_dir():
        raise FileNotFoundError(f'--lcb_runner_path 不是 LiveCodeBench 仓库根目录: {runner_path}')
    print('[LiveCodeBench] 即将调用官方 evaluator 执行模型生成的 Python 代码。')
    subprocess.run(
        build_livecodebench_evaluator_command(args),
        cwd=str(runner_path) if runner_path else None,
        check=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Instinct模型推理、对话与评测")
    parser.add_argument('--benchmark', default='none', choices=['none', 'livecodebench'], help="评测模式（默认进入交互推理）")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--config_path', default=None, type=str, help="原生torch权重对应的模型config JSON（优先于架构CLI参数）")
    parser.add_argument('--model_architecture', default='standard', choices=['standard', 'linear', 'looped'], help="原生torch模型主干")
    parser.add_argument('--residual_type', default='standard', choices=['standard', 'mhc', 'attnres'], help="残差拓扑")
    parser.add_argument('--hc_mult', default=4, type=int, help="mHC并行残差流数量")
    parser.add_argument('--hc_sinkhorn_iters', default=20, type=int, help="mHC Sinkhorn迭代次数")
    parser.add_argument('--attnres_variant', default='block', choices=['full', 'block'], help="AttnRes变体")
    parser.add_argument('--attnres_block_size', default=2, type=int, help="Block AttnRes块大小（按子层计）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0表示贪心生成）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--early_exit', default=0, type=int, choices=[0, 1], help="启用动态Early Exit推理（0=否，1=是）")
    parser.add_argument('--exit_threshold', default=0.9, type=float, help="退出置信度阈值（0-1，默认0.9，仅--early_exit 1时生效）")
    parser.add_argument('--logit_lens', default=0, type=int, choices=[0, 1], help="启用Logit Lens逐层解释，展示每个Transformer层对下一个token的预测（0=否，1=是）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")

    lcb = parser.add_argument_group('LiveCodeBench code generation')
    lcb.add_argument('--lcb_release_version', default='release_latest', type=str, help="数据版本，如 release_v5、release_v6 或 release_latest")
    lcb.add_argument('--lcb_dataset_path', default=None, type=str, help="可选的本地 LiveCodeBench JSON/JSONL，跳过 Hugging Face 下载")
    lcb.add_argument('--lcb_start_date', default=None, type=str, help="仅评测该日期及之后的题目（YYYY-MM-DD）")
    lcb.add_argument('--lcb_end_date', default=None, type=str, help="仅评测该日期及之前的题目（YYYY-MM-DD）")
    lcb.add_argument('--lcb_limit', default=0, type=int, help="仅生成前 N 题（0=全部；smoke test 用，不能直接官方评分）")
    lcb.add_argument('--lcb_num_samples', default=1, type=int, help="每题生成样本数；计算 pass@5 时至少设为 5")
    lcb.add_argument('--lcb_seed', default=42, type=int, help="可复现生成的基础随机种子")
    lcb.add_argument('--lcb_prompt_style', default='auto', choices=['auto', 'chat', 'base'], help="chat 使用 tokenizer 对话模板，base 使用纯文本提示")
    lcb.add_argument('--lcb_output', default='out/livecodebench_generations.json', type=str, help="官方 custom evaluator 格式的输出 JSON")
    lcb.add_argument('--lcb_resume', default=1, type=int, choices=[0, 1], help="从已有输出续跑（每完成一题原子保存）")
    lcb.add_argument('--lcb_evaluate', action='store_true', help="生成后调用官方 evaluator（会执行模型生成的代码）")
    lcb.add_argument('--lcb_evaluate_only', action='store_true', help="跳过模型加载，仅对 --lcb_output 调用官方 evaluator")
    lcb.add_argument('--lcb_runner_path', default=None, type=str, help="LiveCodeBench 官方仓库根目录；已安装 lcb_runner 时可省略")
    lcb.add_argument('--lcb_num_process_evaluate', default=4, type=int, help="官方 evaluator 并发进程数")
    lcb.add_argument('--lcb_timeout', default=6, type=int, help="官方 evaluator 单测试超时秒数")
    return parser


def main():
    args = build_parser().parse_args()

    if args.lcb_num_samples < 1:
        raise ValueError('--lcb_num_samples 必须至少为 1')
    if args.lcb_limit < 0:
        raise ValueError('--lcb_limit 不能为负数')
    if args.benchmark == 'livecodebench' or args.lcb_evaluate or args.lcb_evaluate_only:
        if args.lcb_evaluate_only:
            run_livecodebench_evaluator(args)
            return
        model, tokenizer = init_model(args)
        run_livecodebench_generation(args, model, tokenizer)
        if args.lcb_evaluate:
            run_livecodebench_evaluator(args)
        return
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        if args.logit_lens:
            logit_lens_explain(model, tokenizer, inputs["input_ids"], inputs["attention_mask"])
            conversation.append({"role": "assistant", "content": "(logit lens)"})
            print('\n')
            continue

        print('🧠: ', end='')
        st = time.time()
        generation_kwargs = _generation_kwargs(args, tokenizer)
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            streamer=streamer, **generation_kwargs
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
