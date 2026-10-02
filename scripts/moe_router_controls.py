"""Shared, torch-free routing controls for direct and pipeline launches."""


def migration_args(state, model, trainer, *, from_resume=False):
    if (from_resume or trainer not in ('pretrain', 'cpt', 'full_sft')
            or not model.get('use_moe') or not state.get('moe_router_migration', False)):
        return {}
    experts = int(model.get('num_experts', state.get('num_experts', 4)))
    top_k = int(model.get('num_experts_per_tok', state.get('num_experts_per_tok', 1)))
    if not 1 <= top_k <= experts:
        raise ValueError(f'迁移 top-k 必须在 1 到专家总数 {experts} 之间，当前为 {top_k}')
    normalize = bool(model.get('norm_topk_prob', state.get('norm_topk_prob', top_k > 1)))
    return {
        'moe_router_top_k': top_k,
        # Normalizing a single selected probability removes the task gradient.
        'moe_router_norm_topk_prob': int(normalize and top_k > 1),
    }
