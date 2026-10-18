"""CPU-only presentation helpers for the training configuration workspace."""
from html import escape


def workspace(st):
    st.markdown("""
<style>
    .stMainBlockContainer { max-width: 1600px; padding-top: 4rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] .stSidebarBlockContainer { padding-top: 2rem; }
    .workspace-eyebrow { color: #93c5fd; font-size: 12px; letter-spacing: .16em; font-weight: 650; }
    .workspace-title { font-size: clamp(26px, 3vw, 36px); font-weight: 750; letter-spacing: -.04em; margin: 3px 0; }
    .workspace-subtitle { color: #a5b4c9; font-size: 14px; margin-bottom: 16px; line-height: 1.7; }
    .workspace-summary { display: grid; grid-template-columns: 1.5fr repeat(3, 1fr); gap: 12px; margin: 0 0 18px; }
    .workspace-card { background: #162135; border: 1px solid #2b3b53; border-radius: 12px; padding: 14px 18px; min-width: 0; }
    .workspace-card-label { color: #a5b4c9; font-size: 12px; margin-bottom: 6px; }
    .workspace-card-value { color: #f1f5f9; font-size: 20px; font-weight: 650; overflow-wrap: anywhere; }
    .workspace-card-detail { color: #94a3b8; font-size: 12px; margin-top: 5px; overflow-wrap: anywhere; }
    [data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: 1px solid #334155; margin-bottom: 16px; overflow-x: auto; }
    [data-testid="stTabs"] [role="tab"] { padding: 10px 16px; height: auto; white-space: nowrap; border-radius: 8px 8px 0 0; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: #1e3354; color: #bfdbfe; }
    [data-testid="stExpander"] { border-radius: 12px; border-color: #334155; }
    [data-testid="stExpander"] summary { padding-top: 14px; padding-bottom: 14px; }
    [data-testid="stExpander"] summary p { font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stButton button, .stDownloadButton button { border-radius: 8px; min-height: 40px; }
    @media (max-width: 900px) {
        .workspace-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .stMainBlockContainer { padding-left: 1rem; padding-right: 1rem; }
        [data-testid="stTabs"] [role="tab"] { padding: 10px 12px; }
    }
    @media (max-width: 480px) {
        .workspace-summary { grid-template-columns: 1fr; }
        .workspace-card { padding: 12px 14px; }
    }
</style>
<div class="workspace-eyebrow">INSTINCT / TRAINING WORKSPACE</div>
<div class="workspace-title">训练工作台</div>
<div class="workspace-subtitle">配置模型与数据，实测性能，再启动训练。参数修改自动保存。</div>
""", unsafe_allow_html=True)
    summary = st.container()
    activity = st.container()
    # Render every tab on each run so moving between workspaces never deletes
    # Streamlit widget state or bypasses training/profile validation.
    train, tune, model, pipeline, monitor, export = st.tabs([
        "训练配置", "自动调优", "模型结构", "训练流水线", "训练监控", "配置导出",
    ])
    with train:
        st.caption("01 选择训练任务与数据 → 02 调整批次与性能 → 03 检查命令并启动。需要实测推荐时，进入「自动调优」。")
        launch = st.container()
        train_left, train_right = st.columns(2, gap="large")
    with model:
        st.caption("左侧选择预设；下方修改结构参数。修改会同步更新参数量、架构图和训练命令。")
        model_left, model_right, diagram = st.columns([1, 1, 1.15], gap="large")
        breakdown = st.container()
    with pipeline:
        st.caption("分别保存 Pretrain 与 SFT 配置快照，再运行完整训练流程。")
    with monitor:
        st.caption("查看当前训练状态、日志和历史曲线对照。")
    return dict(summary=summary, activity=activity, train_left=train_left,
                train_right=train_right, launch=launch, tune=tune,
                model_left=model_left, model_right=model_right, diagram=diagram,
                breakdown=breakdown, pipeline=pipeline, monitor=monitor, export=export)


def render_summary(st, cfg, breakdown):
    state = st.session_state
    total = breakdown['Total Params']['value']
    active = breakdown.get('Active Total', {}).get('value', total)
    stage = state.get('train_type', 'pretrain')
    packing = state.get('sequence_packing') and state.get('sequence_packing_mode') == 'bucket'
    if stage == 'pretrain' and state.get('autotune_use_profile'):
        policy, detail = '调优 profile', '以命令预览与启动校验为准'
    elif packing:
        budget = state.get('bucket_token_budget', 0)
        policy = f'{budget:,} tokens/卡' if budget else '动态显存预算'
        detail = '续训沿用保存预算' if state.get('from_resume') else '按桶长计算 batch；启动时校准'
    else:
        policy, detail = f"Batch {state.get('batch_size', 1)}", '固定批量 · 可在训练配置中调整'
    cards = [
        ('当前模型', state.get('preset', 'Custom'), f"{cfg['num_hidden_layers']} 层 · hidden {cfg['hidden_size']}"),
        ('参数量 / 每 token 激活', f'{total / 1_000_000:,.1f}M', f'激活 {active / 1_000_000:,.1f}M'),
        ('训练阶段', stage.upper(), '断点续训' if state.get('from_resume') else '新训练阶段'),
        ('批量策略', policy, detail),
    ]
    html = ''.join(
        '<div class="workspace-card">'
        f'<div class="workspace-card-label">{escape(label)}</div>'
        f'<div class="workspace-card-value">{escape(str(value))}</div>'
        f'<div class="workspace-card-detail">{escape(detail)}</div></div>'
        for label, value, detail in cards
    )
    st.markdown(f'<div class="workspace-summary">{html}</div>', unsafe_allow_html=True)
