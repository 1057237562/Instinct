# 开发 Workflow 强制约束

本文档是 workflow 的**入口索引**：只保留硬门禁与路由指示。所有 Phase 细则由 `docs/ai-rules/task-workflow.md` 承载（该文件按需读取，不进每次会话基线）。

---

## 工作流程总览

```
收到需求
  │
  ▼
Phase 0: 知识库检索
  │
  ▼
Phase 1: PRD 准入评审 ──── 严重缺陷 ──→ HOOK①: 人工判断 ──→ 驳回 / 放行
  │                                                            │
  │ (通过)                                                      │
  ▼                                                            ▼
Phase 2: 需求澄清 ←──────────────────────────────── (放行后继续)
  │
  │ (零疑点)
  ▼
Phase 3: Plan 方案设计
  │  3.0 创建 SPEC-xxx + 四件套 + repos.json
  │  3.1 requirements.md（产运语言）
  │  3.2 plan.md（FA→TC 1:N 覆盖矩阵）
  │  3.3 feature_list.json（acceptanceCriteria 反向绑定 FA/TC）
  │  3.4 Review 前跑 TC 双向 diff 自检
  ▼
Phase 3.5: HOOK②: 方案 Review（产运 + 研发双 ack）
  │
  │ (双方 ack)
  ▼
Phase 4: TDD 实现（委托 everything-claude-code:tdd-workflow skill）
   对每个 feature 的每条 TC: Red(活证明) → Green → Refactor → 回写 evidence
   子仓在 git 下 → 每阶段 checkpoint commit，evidence 带 sha
   hook 自动写 progress/<session_id>.jsonl，不手工维护
```

---

## Phase 硬门禁（详细规则见 `docs/ai-rules/task-workflow.md`）

### Phase 0 知识库检索
**触发**：收到需求后、任何阶段遇到疑点时。
**动作**：先查 `~/.claude/knowledge/`、项目 `docs/domain/`、`~/.claude/plans/`、项目代码历史。查得到的不问；查不到的标"已知 + 待澄清"。

### Phase 1 PRD 准入评审
按 5 维度评审（业务完整性 / 功能完整性 / 逻辑一致性 / 可实现性 / 验收标准）。PASS/WARN 直通 Phase 2；**BLOCK 必须 HOOK 人工判断，不得自行降级**。详见 `task-workflow.md §Phase 1`。

### Phase 2 需求澄清
12 类别全检（需求细节 / 领域术语 / 代码上下文 / 技术实现 / 数据流 / 测试方法 / 测试覆盖面 / **UX 视觉契约（FE 必查 PRD 截图）** / **跨字段联动状态机（表单类必画真值表）** / **接口字段契约盲核（直接读后端 DTO 源码）** / 灰度回滚 / 影响面），零疑点才能退出。**退出条件不满足禁止进入 Phase 3**。详见 `§Phase 2`。

### Phase 3 方案设计
- 3.0 创建 `docs/specs/SPEC-xxx/` + 四件套 + `repos.json`
- 3.1 `requirements.md`（产运语言）+ `plan.md`（**必须用 FA→TC 1:N 覆盖矩阵组织 Verification Plan**）
- 3.2 `feature_list.json` 初始化：每 F 带 `acceptanceCriteria` 反向绑定 FA/TC
- 3.3 Review 前跑 TC 双向 diff 自检（两次 diff 空集才通过）
- 详见 `§Phase 3`

### Phase 3.5 方案 Review HOOK（强制）
产运 ack `requirements.md` + 研发 ack `plan.md` + `feature_list.json`，**两次 TC 双向 diff 空集**。不得自行判定通过。详见 `§Phase 3.5`。

### Phase 4 TDD 实现（委托 `everything-claude-code:tdd-workflow` skill）
- **RED 硬门禁**：编译 + 执行 + 因预期业务原因失败；只写不跑不算 RED，不得进入 Green
- **FE 测试运行时前置检查**：FE 子仓无 jest/vitest 时禁止声明 `*-FE-Unit` TC；二选一硬决策（先配运行时 / 降级 MANUAL）须用户 ack。详见 `task-workflow.md §4.6`
- **BE 集成测试目录约定**：IT 必须落 `src/integration-test/java/`，禁止与 UT 混在 `src/test/java/` + `@Ignore` 切换。详见 `task-workflow.md §4.7`（KM 2378243571）
- **子仓在 git 下**：每 TDD 阶段强制 checkpoint commit，evidence 带短 sha
- **commit 分支可达性**：所有引用的 sha 必须从子仓当前 HEAD 可达（active 分支）
- **passing 充要条件**：`acceptanceCriteria.tc` 所有 TC 在 `evidence` 记 PASS + 所有 sha HEAD 可达
- 覆盖率 soft metric，不强制
- 详细执行规则（evidence 格式 / 状态流转 / fullstack 拆分惯例 / FE 运行时门 / BE IT 目录约定）见 `task-workflow.md §4.1-§4.7`

---

## ⚠️ AGENT TOOL ENVIRONMENT 强约束（避免 sub-agent 选错 JDK / 改测试框架）

**背景**：Claude Code sub-agent 的 bash 子进程**不继承用户 `.bash_profile` / `.zshrc` 中的 JAVA_HOME**。在 macOS 上 `mvn` 会走 `/usr/libexec/java_home` 默认（首个 JDK，常常是最新的 17/21），与项目实际所需版本不符，导致编译/测试出现 `endPosTable`、模块访问、ClassCastException 等错误。

**已知错误响应模式**（必须避免）：sub-agent 把 JDK 17 incompatibility 当成"项目 bug"，自行切换 JDK / 替换测试框架（如把 `mockito-core + powermock` 换成 `mockito-inline`）/ 改 Lombok 版本来"修复"环境问题 — 这违反"不动业务依赖"的硬约束。

### 强约束规则

1. **任何 mvn / gradle / java 命令**前，必须先从 `docs/specs/<SPEC>/repos.json[<repo>].tool_env.java_home` 读到对应 JDK 路径，并显式 `export JAVA_HOME=<path>`（或 inline 一次性 `JAVA_HOME=<path> mvn ...`）。

2. **DO NOT 改 JDK 版本**：项目声明的 `tool_env.jdk_major` 是权威。如果本机没装该版本，**报错并停止**，不得自行切换到其它 JDK。

3. **DO NOT 改测试框架版本/类型**：mockito-core / powermock-api-mockito2 / powermock-module-junit4 / junit 等版本已被项目 pom 锁定，**不许自行替换**（如 mockito-inline 替代 powermock）来绕过环境问题。

4. **DO NOT 改 Lombok 版本**：Lombok 与 JDK 强耦合（如 1.18.20 兼容 JDK 8/11，不兼容 JDK 17/21）。**遇到 Lombok 报错先排查 JAVA_HOME，不动 Lombok 版本**。

5. **遇到环境错误的诊断顺序**（强制）：
   - **Step 1**：`mvn -version` 看实际 Java 版本，对照 `repos.json[].tool_env.jdk_major`
   - **Step 2**：如果 mismatch → 显式 `export JAVA_HOME=<repos.json 里 java_home>` 后重试
   - **Step 3**：如果仍报错 → **停下来汇报给人**，不得自行修改 pom/dep

### Agent 调用 mvn 的正确姿势

```bash
# 1. 从 repos.json 取 JAVA_HOME
WORKSPACE_ROOT="<harness-cwd>"
SPEC="SPEC-xxx"
REPO_NAME="credit-kylin"
JAVA_HOME=$(jq -r ".repos[]|select(.name==\"$REPO_NAME\").tool_env.java_home" \
    "$WORKSPACE_ROOT/docs/specs/$SPEC/repos.json")

# 2. 显式 export，再跑 mvn
export JAVA_HOME
cd /path/to/$REPO_NAME
mvn -pl <module> -am test -Dtest=<TestClass>
```

**不要**这样写（继承 shell 默认 JAVA_HOME，会落到错版本）：
```bash
cd /path/to/repo && mvn test  # ✗ 不显式 export JAVA_HOME
```

### 派 sub-agent 任务时的强提示模板

派给 sub-agent 的 prompt **必须包含**这段：

> "You MUST read `docs/specs/<SPEC>/repos.json[<repo>].tool_env.java_home` and `export JAVA_HOME=<that_path>` before any mvn/gradle command. DO NOT change JDK version, Lombok version, or test framework dependencies (mockito-core, powermock, junit) to 'fix' tool errors — if mvn fails, FIRST verify JAVA_HOME matches `tool_env.jdk_major`, then if still failing, STOP and report instead of muddling through."

---

## ECC 插件依赖

Phase 4 TDD 强依赖 ECC 插件。检测命令：

```bash
jq -e '.plugins["everything-claude-code@everything-claude-code"]' \
   ~/.claude/plugins/installed_plugins.json
```

未装时 `init.sh` 结尾已告警；补装方式（在 Claude Code 对话框里输入，不是 shell）：

```
/plugin marketplace add https://github.com/affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

---

## 全局硬规则

1. 任何阶段遇到疑点 → 先 Phase 0 检索（知识库、代码、memory），不问已知
2. 不得跳过 Phase 1 和 Phase 2，即使用户说"直接做"也要先过一遍
3. Phase 1 BLOCK 级别缺陷必须 HOOK 人工判断，不可自行降级
4. Phase 2 退出条件是硬门槛，不是建议
5. Phase 3 必须同时产出 `requirements.md` + `plan.md` + `feature_list.json`；plan 用 FA→TC 1:N 组织；feature_list 的 `acceptanceCriteria` 与 plan 的 FA/TC 双向绑定；Review 前过双向 diff 自检
6. Phase 3 → Phase 4 必须经 Phase 3.5 人工 Review HOOK
7. 实现中发现 PRD 未覆盖的场景 → 停下来回 Phase 0/2 补澄清，不自行补全逻辑
8. 先写测试再写实现（TDD），**RED 必须是活的**（编译 + 执行 + 正确失败）
9. feature 进入 `passing` 必须所有 `acceptanceCriteria.tc` 在 `evidence` 记 PASS；子仓在 git 下时所有 commit sha 必须从 HEAD 可达
10. Phase 4 的 Red/Green/Refactor **必须委托 `everything-claude-code:tdd-workflow` skill** 执行

---

## 知识库路径

| 位置 | 内容 | 加载方式 |
| :--- | :--- | :--- |
| `~/.claude/knowledge/` | 全局知识库（跨项目） | Phase 0 按需读取 |
| 项目 `docs/domain/` | 项目级知识库（团队共享） | Phase 0 按需读取 |
| `docs/ai-rules/task-workflow.md` | **Workflow 细则**（Phase 0-4 + Spec 四件套） | 进入对应 Phase 时按需读取 |

---

## 多仓 Harness 上下文（由 multirepo-init skill 安装）

本工作空间是**多仓 harness**：本身不含业务源码，承担需求文档、跨仓协调、SPEC 四件套维护职责。

### 目录约定

| 路径 | 内容 |
| :--- | :--- |
| `docs/specs/SPEC-xxx-name/` | 当前 harness 维护的 SPEC（一需求一目录） |
| `docs/specs/SPEC-xxx-name/repos.json` | manifest：声明本 SPEC 涉及的子仓清单（仅 harness 本地） |
| `docs/specs/SPEC-xxx-name/progress/<sid>.jsonl` | 本会话 turn 流水（hook 权威写入；sync-specs 镜像到各子仓） |
| `docs/specs/_unassigned/progress/<sid>.jsonl` | SPEC 未绑定前的临时 turn，绑定后自动回填到目标 SPEC |
| `docs/ai-rules/task-workflow.md` | Workflow 细则（Phase 0-4 + Spec 四件套规范） |
| `.claude/hooks/progress.sh` | Hook：每轮 Stop 把本 turn finalize 成一条 jsonl |
| `.claude/hooks/sync-specs.sh` | Hook：Stop 后把 harness 的四件套 + progress/ 镜像到各子仓 |

### Phase 3 创建 SPEC 时的多仓特性

- 创建 `SPEC-xxx/` 后**必须同时创建 `repos.json`** 声明涉及子仓；路径必须是绝对路径
- `repos.json` 缺失 / 留空 → turn 数据落 `_unassigned/`，子仓侧拿不到 telemetry

### Phase 4 实现时的多仓特性

- progress.sh 在 Stop 时 finalize turn，写 harness `progress/`
- sync-specs 立刻把 harness 的 `progress/` + 四件套镜像到每个子仓 `docs/specs/<SPEC>/`（byte-for-byte 一致）
- 子仓独立打开 Claude 时也能看完整 turn 历史 + SPEC 文档
- 不要手工维护子仓 `.claude/`

### sync-specs 范围

- **同步**：四件套 + 整个 `progress/`
- **不同步**：`repos.json`（harness 内部 manifest）、SPEC 根下其它顶层文件（除非加进 sync-specs 白名单）
- **不碰子仓 git**：sync 只 cp 文件，不 add / 不 commit

手动强制 sync：`.claude/hooks/sync-specs.sh "$(pwd)" [<SPEC-ID>]`

### 已知约束

- hook 只在 harness `.claude/settings.json` 注册，不接管子仓 `.claude/`
- 改 hook 后**必须重启 Claude Code 会话**才生效（settings 启动时加载）
- 子仓独立开 Claude 时走子仓自己的 hook（若有），互不干扰