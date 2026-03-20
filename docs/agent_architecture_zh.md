# ScryNeuro Agent 架构详解（面向科研与可扩展）

本文档面向以下目标：

- 让初学者理解 ScryNeuro Agent 的整体结构与运行方式
- 解释为什么这样设计（设计思路与取舍）
- 说明如何在不破坏核心的前提下扩展工具、技能、插件
- 给出与 pi-mono 这类「核心 loop 导向」项目的对比

---

## 1. 一句话定位

ScryNeuro Agent 是一个 **Prolog 控制面 + Python 执行面** 的神经符号 Agent 框架：

- Prolog 负责可解释的搜索/逻辑编排
- Python 负责 LLM、工具执行、技能注入、插件钩子、会话与日志
- Rust/PyO3 负责 Prolog 与 Python 的高性能桥接

这意味着：你可以在 Prolog 里清晰表达「搜索树与策略」，又能复用 Python 生态完成复杂动作。

---

## 2. 分层架构（从用户到运行时）

```text
┌────────────────────────────────────────────────────────────┐
│ 用户 Prolog 代码（研究逻辑/搜索树）                       │
│  examples/real_llm_agent.pl 等                            │
└───────────────────────┬────────────────────────────────────┘
                        │ 调用 facade API
┌───────────────────────▼────────────────────────────────────┐
│ Prolog Facade（用户友好 API）                             │
│  prolog/scryer_agent_api.pl                               │
│  - agent_new_from_profile/2,3                             │
│  - agent_enable_tools/2, agent_enable_skills/2            │
│  - agent_run/3, agent_trace/2, agent_close/1              │
└───────────────────────┬────────────────────────────────────┘
                        │ 调 core plugin
┌───────────────────────▼────────────────────────────────────┐
│ Prolog Core Plugin（桥接与句柄管理）                      │
│  prolog/scryer_agent.pl                                   │
│  prolog/scryer_tool_predicates.pl                         │
└───────────────────────┬────────────────────────────────────┘
                        │ FFI 调 Python
┌───────────────────────▼────────────────────────────────────┐
│ Python Runtime（真实 agent 逻辑）                         │
│  python/scryer_agent/runtime.py                           │
│  python/scryer_agent/tool_runtime.py                      │
│  python/scryer_llm_runtime.py                             │
│  python/scryer_agent/config.py                            │
└────────────────────────────────────────────────────────────┘
```

### 为什么分层？

1. **认知分离**：Prolog 只看策略与流程，不被大量提示词胶水污染。  
2. **演化分离**：LLM/tool/skill 变化快，放 Python 易迭代；逻辑层保持稳定。  
3. **科研可控**：核心循环可追踪（trace + JSONL），便于复现实验。

---

## 3. 核心模块地图（按职责）

### 3.1 用户 API 层

文件：`prolog/scryer_agent_api.pl`

主要谓词（常用）：

- `agent_providers/1`：查看支持的 provider
- `agent_profiles/1` / `agent_profile/2`：查看配置 profile
- `agent_new_from_profile/2,3`：按 profile 创建 agent
- `agent_enable_tools/2`：启用内置工具
- `agent_enable_skills/2`：启用技能
- `agent_run/3`：运行自动 loop
- `agent_trace/2`：读取运行轨迹
- `agent_close/1`：卸载 agent

### 3.2 Prolog 核心桥接层

文件：`prolog/scryer_agent.pl`

作用：

- 管理 Prolog 侧 `agent_registry`（名字到 Python 句柄）
- 把 Prolog options 转为 Python kwargs
- 调用 Python runtime 的模块函数

文件：`prolog/scryer_tool_predicates.pl`

作用：

- 让工具可被 Prolog 直接调用（神经谓词化）
- 如 `tool_shell_exec/...`、`tool_call_json/3`

### 3.3 Python Agent Runtime 层

文件：`python/scryer_agent/runtime.py`

核心能力：

- `AgentManager.create/step/run/trace/unload`
- 技能解析与注入（`SKILL.md` frontmatter）
- 插件钩子（plan/tool/step 前后）
- 会话保存/恢复
- 实验日志 JSONL

### 3.4 Python Tool Runtime 层

文件：`python/scryer_agent/tool_runtime.py`

核心能力：

- 工具注册（entrypoint = `module:function`）
- 内置工具目录（`web_fetch/shell_exec/read_file/...`）
- 统一调用与结构化结果

### 3.5 LLM Runtime 层

文件：`python/scryer_llm_runtime.py`

核心能力：

- 统一 provider 接口：`openai/anthropic/huggingface/ollama/custom`
- 同一 generate 路径，按 provider 分派

### 3.6 配置系统层

文件：`python/scryer_agent/config.py`

核心能力：

- 优先读取 `python/scryer_agent/config/agent_profiles.json`
- 兼容读取旧路径 `python/config/agent_profiles.json`
- 可选读取 `.local.json` 并深度合并
- 按 profile + override 解析最终参数

---

## 4. Agent 运行流程（端到端）

## 4.1 创建阶段

1. Prolog 调 `agent_new_from_profile(Name, Profile, Options)`
2. Python 读取 profile 并合并覆盖项
3. 解析 provider/model/key/base_url
4. 初始化 `AgentState`（messages/tools/skills/trace/logger）

## 4.2 运行阶段（step/run）

一次 step 大致过程：

1. 构建 action prompt（含工具描述、技能片段、历史消息）
2. 让模型返回 action JSON（`respond` 或 `tool_call`）
3. 若 `tool_call`：执行工具，写回 observation
4. 继续直到 `done` 或到达阈值（如 max_auto_tools）

`run` 则是多次 step，直到 `done=true` 或 `max_steps`。

## 4.3 观测与复现

- `agent_trace/2`：内存中的逐步轨迹
- JSONL 实验日志：`run_start/tool_call/step_end/...`
- 会话保存恢复：便于中断后复跑

---

## 5. 设计思路（为什么这么做）

### 5.1 设计原则 A：核心 loop 简洁

核心 loop 只做四件事：

- 计划（plan）
- 行动（tool/response）
- 观察（observation）
- 继续/终止（continue/stop）

这样便于科研中做 ablation（替换某一步并比较效果）。

### 5.2 设计原则 B：可插拔而非硬编码

- 工具：注册即用（entrypoint）
- 技能：按需注入（frontmatter + policy）
- 插件：生命周期钩子

优点：同一框架可服务不同实验主题（代码代理、网页检索、符号推理）。

### 5.3 设计原则 C：对 Prolog 用户友好

用户看到的是简洁 API 与搜索流程，而不是提示词模板细节。  
提示词胶水、provider 细节、技能注入逻辑都隐藏在 Python runtime。

### 5.4 设计原则 D：可审计、可复现

实验日志 schema + trace + session checkpoint 三件套，保证：

- 能回放
- 能比较
- 能定位回归问题

---

## 6. 配置系统与优先级

配置来源按优先级（高到低）：

1. 调用时显式 options
2. profile 字段（优先本地 `agent_profiles.json`；若缺失则回退到 `agent_profiles.example.json`；再叠加 `.local`）
3. 环境变量/.env（例如 `OPENAI_MODEL`）
4. 代码硬默认

对研究来说，这个顺序能兼顾：

- 团队共享默认配置（profile）
- 个人本地私有覆盖（local/.env）
- 实验脚本中的一次性 override

---

## 7. 与 pi-mono 的对比（核心导向视角）

说明：以下对比聚焦「核心 agent loop 与科研可扩展性」，不是做产品体验排名。

| 维度 | ScryNeuro | pi-mono（核心导向） |
|---|---|---|
| 目标定位 | 神经符号研究、Prolog+Python 混合控制 | 极简核心循环、单体高聚合实现 |
| 控制面 | Prolog（逻辑/搜索树） | 通常在单一运行时内表达控制逻辑 |
| 执行面 | Python runtime（LLM/工具/技能/插件） | 核心 loop 驱动工具与模型调用 |
| 扩展方式 | 工具注册 + 技能注入 + 插件钩子 + profile | 强调核心最小闭环，扩展通常围绕 loop 机制 |
| 可观测性 | trace + JSONL schema + session | 常强调 loop 清晰与状态可追踪 |
| 会话管理 | 支持保存/恢复，适合实验复跑 | 强调会话持久化与分支探索 |
| 工具触发模式 | 由运行时 action JSON 驱动工具调用 | LLM 在 loop 中触发工具调用 |
| 复杂度取舍 | 跨语言能力强，但系统复杂度更高 | 结构更轻量，迁移理解成本更低 |
| 研究友好性 | 适合把符号搜索与 LLM 行为并行研究 | 适合快速迭代核心 loop 策略 |

### 7.1 ScryNeuro 相对优势

1. **神经符号天然契合**：Prolog 搜索树表达能力强。  
2. **扩展面更宽**：tools/skills/plugins/profile 都是模块化入口。  
3. **实验治理更系统**：日志 schema 与会话恢复是内建能力。

### 7.2 相对劣势/成本

1. **系统复杂度更高**：跨语言（Prolog/Rust/Python）调试成本高。  
2. **部署约束更多**：依赖链（PyO3/Python 版本/provider SDK）更敏感。  
3. **默认 loop 还需持续打磨**：在复杂任务上的策略深度仍可继续增强。

### 7.3 “向 pi-mono 靠拢”在本项目里意味着什么

不是抄外形，而是收敛到这三个核心：

1. **最小且稳定的 action loop**（plan → act → observe → continue）
2. **严格的动作协议**（JSON action schema、错误可诊断）
3. **把增强能力当插件**（不要让核心 loop 失控膨胀）

### 7.4 实践对齐建议（可直接落地）

1. **收敛默认工具集**：默认只启最小工具，其他按任务启用。  
2. **缩短默认系统提示**：把强约束做成协议（schema + 校验），不是堆长提示词。  
3. **强化失败恢复路径**：tool 失败时固定走“诊断→重试→降级”的统一链路。  
4. **保持 loop 核心稳定**：新增能力优先走插件钩子，不直接侵入 step/run 主链。

---

## 8. 当前已具备能力与短板

### 已具备

- Facade API（用户面简化）
- Profile 配置体系（含 local override）
- 工具注册与目录
- 技能发现/注入（frontmatter）
- 插件钩子、会话、实验日志

### 仍可加强（建议路线）

1. 更强的 planner/反思机制（多步规划与失败恢复）
2. 更细粒度的 memory 压缩策略（按任务类型）
3. 更严格的工具 I/O schema 校验与回放测试
4. 更系统的 benchmark 场景（代码、网页、符号推理、RL 交叉）

---

## 9. 给用户的最小实践模板

```prolog
:- use_module('../prolog/scryer_py').
:- use_module('../prolog/scryer_agent_api').

main :-
    py_init,
    agent_new_from_profile(research_agent, "default_openai"),
    agent_enable_tools(research_agent, [web_fetch, shell_exec, write_file]),
    agent_enable_skills(research_agent, ["research-web-markdown"]),
    agent_run(research_agent, "抓取一个网页并整理成 markdown", Out),
    format("~s~n", [Out]),
    agent_trace(research_agent, Trace),
    format("~s~n", [Trace]),
    agent_close(research_agent),
    py_finalize.

:- initialization(main).
```

---

## 10. 结语

ScryNeuro Agent 的核心价值不在“功能堆叠”，而在于：

- 用 Prolog 保留可解释的符号控制
- 用 Python 承担快速演化的神经执行
- 用模块化边界保证长期可扩展与可复现

这条路线非常适合你当前的 neuro-symbolic + RL 科研目标：核心 loop 稳定，能力通过工具/技能/插件持续外接。
