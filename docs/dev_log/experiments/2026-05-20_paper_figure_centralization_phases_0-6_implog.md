# 论文图表中央化 Phase 0–6 实施日志

> **性质**:这是一份*实施日志*(implementation log),记录"实际做了什么 + 为何偏离原计划",非 forward plan。
> **原始逐字计划**:Phase 0–6 的完整逐字原始计划见会话 transcript
> `C:\Users\zhang\.claude\projects\c--Users-zhang-Desktop-github-EEG-BCI\25f72a51-813a-4970-aa47-25d2575469e8.jsonl`(如需绝对保真)。
> **后续工作**:Phase 7+(剩余 6 条人工评论修复)见计划文件
> `C:\Users\zhang\.claude\plans\centralize-all-of-the-synchronous-hennessy.md`。
> **归档日期**:2026-05-20。

## 目标

论文图表(paper figure)的唯一生成入口 + 持久版本链 + staging UI + draft 图回溯审计。
解决四个痛点:生成入口分散、版本历史碎片化、draft 标注可能过期、skill 设计与需求不匹配。

## 总览结果

Phase 0–6 全部完成。

- **Phase 5/6** 提交 `d747306`(8 files, +907 / −38)。
- **Phase 0–4** 提交于 `d747306` 之前的序列(`import-snapshots`、skill server/cli/UI、Phase 3.7/3.8 UI 重设计)。
- 最近相关提交链(`git log --oneline`):
  `d747306 feat(paper-figures): centralize generation + registry-driven version mgmt (Phase 5/6)`,
  `58a27e2`、`23a6790`、`87d86f3`、`6cdb26c`。

## 逐阶段 Planned → As-Built

### Phase 0 — 冻结现状

- **Planned**:git snapshot + copy 14 原图 + `claims.json` + `figure_registry` 骨架,纯增量到 `paper/figures/_audit_corpus/2026-05-13/`,绝不修改既有 script/draft/doc。
- **As-Built**:完成。范围实际为 **21** 张 registry 图(非最初估的 14)。`_audit_corpus/2026-05-13/` 冻结隔离,后续 Phase 改动脚本不影响此快照。
- **偏差**:图数 14 → 21。

### Phase 1 — 审计重生成

- **Planned**:逐图跑当时 working tree 里**未修改**的 claimed 命令,结果 `copy2` 到 `regenerated/`,跑完用 `originals/` 复原 `paper/figures/`(净影响 = 0)。
- **As-Built**:完成。multi-source 候选记录在 `audit_report`;命令失败的图标 `failed=true` 不中断批次。
- **偏差**:无重大偏差。

### Phase 2 — registry 补全 + 存储迁移

- **Planned**:回填权威命令字段;建 `_history/<fig_id>/manifest.json` + `_index.json`;`import-snapshots` 子命令导历史快照 + 审计 staging。
- **As-Built**:完成。manifest schema = `trunk[]`(已接受 vN)/ `staging[]`(待裁决 sN)/ `rejected[]`(软删 rN);评论按 `(before_sha, after_sha)` 内容寻址,跨 accept/reject 存活。
- **偏差**:无重大偏差。

### Phase 3 — skill 改造(history mode + staging UI)

- **Planned**:`history_cli.py` 全子命令 + `history_server.py`(stdlib http.server,127.0.0.1:8765)+ vanilla-JS web UI + schema 文档;`build_compare_page.py` 标 deprecated。
- **As-Built**:完成。**额外新增 Phase 3.7 / 3.8**:UI 三栏重设计 + History modal + Copy-ref 按钮(均纯前端);`comment-add` / `comment-at-tip` / `comment-status` / `comments-open` / `context-bundle` 子命令补齐。
- **偏差**:范围扩展(Phase 3.7/3.8 为计划外增量,提升可用性)。

### Phase 4 — 用户 UI 裁决

- **Planned**:用户对 staging 视觉 Accept/Reject + 多源选权威。
- **As-Built**:完成。用户做完全部 accept/reject,**留下 6 条 plot-fix 评论**(fig1 / fig2 / fig6b / fig8 / fig10b / fig_s1)——即 Phase 7 的输入。fig2 / fig3c 走 **Option A**(接 bug-fix native 生成器并 stage),两者 bug-fix native v2 均被 Accept。
- **偏差 / 遗留**:fig2 / fig3c v2 虽被 Accept,但**顶部 pane 标题缺陷未真正修复**(根因在未提交的 `comparison.py` hunk)→ 移交 Phase 7 A1。

### Phase 5 — 中央化 `generate_paper_figures.py`

- **Planned**:加 7 个新 figure target,registry 驱动 dispatch,`--stage-history` 默认开,`--figure all` 全覆盖,`dapt_v1_v5_heatmap` 标 DEPRECATED。
- **As-Built**:完成,提交 `d747306`。实现:
  - `_stage_to_history(spec, produced, source_cmd)` → 返回 `staged` / `deduped` / `error`(grep "dedup"/"byte-identical" 把字节相同归为干净跳过,非失败)。
  - `_generate_one(spec, stage_history)` → native(`figure_generators_key` 设则进程内调用)或 subprocess(`generator_command`);**mtime guard**(对比 run 前 `st_mtime` 快照,检测"跑了但没写"防 stale staging);永不抛异常。
  - 新 `main()`:argparse,`--figure` choices = fig_ids ∪ legacy keys ∪ `all`,`--stage-history` 用 `argparse.BooleanOptionalAction`(default=True);批量任一失败不中断,结尾汇总并非零退出。
  - `dapt_v1_v5_heatmap` 注释改为 `# DEPRECATED (2026-05-12, Stage 4 Step 4): ...`(supplementary 备份,不在 registry,不入 `--figure all`)。
- **偏差**:**未跑 `--figure all --stage-history`**——因当时工作树存在未审 `src/visualization/*` 改动,跑全量会批量 stage 21 张未审图并抢跑 6 条待修评论。**故意推迟到 Phase 7 Workstream C**。

### Phase 6 — draft 路径修正 + 文档

- **Planned(原始)**:盲改 draft 里 `../../results/<timestamp>...png` → `../figures/<canonical>.png`;CLAUDE.md 加段;弃用旧 compare 脚本。
- **As-Built — 重大偏差(已与用户确认走 Option A)**:
  - **发现**:registry + 5 个 history manifest + Phase-5 dispatch + draft 已统一以各图真实 `canonical_output_path` 为键;fig1/2/3c/6/6b 按**设计**就在 `results/<timestamp>...`(timestamped 数据溯源图)。原始"盲改"会**错误破坏**这 5 个 `results/` 引用。
  - **改为**:注册表一致性校验器 `scripts/paper/update_draft_image_paths.py`(report-only 默认,有 MISMATCH 退出 1 = CI 友好;`--apply` 写 `.bak` 后按 registry `canonical_output_path` 规范修正;非 registry 图如电极放置图 3a / S3–S6 报 `NOT_IN_REGISTRY` 跳过)。
  - **实跑结果**:修正 `paper_draft_v3.1.md` 中 8 处真实 mismatch(figs 7/8/9/3b/4/10a/10b/11,`../../paper/figures/X.png` → `../figures/X.png`)→ 21 OK / 0 MISMATCH;备份 `paper/drafts/paper_draft_v3.1.md.bak_20260520_020449`。
  - **文档**:CLAUDE.md 新增 "## 论文图表生成与版本管理" 段;`scripts/paper/build_figures_compare_page.py`、skill 内 `.claude/skills/figure-snapshot-diff/scripts/build_compare_page.py`、根目录 `paper/figures_compare*.html` 加 `# DEPRECATED (2026-05-20, Phase 6)` banner;`SKILL.md` Legacy pair mode 段加 DEPRECATED blockquote + History mode 章节。
  - **决策依据**:Option A(注册表一致性校验)由用户经 AskUserQuestion 明确选定,事后证明正确(捕获盲改会漏的 8 处真实 mismatch,且避免错误破坏 5 个 `results/` 引用)。

## 关键已知坑(Phase 0–6 沉淀,Phase 7 仍适用)

- fig1/2/3c/6/6b 是 timestamped result path,按设计 canonical 在 `results/<timestamp>...`;重生成只换"用什么 plotting 代码渲染**同一份** cached results",**不可**用最新 commit 当新数据,timestamp 必须 hardcode 在 registry。
- `run_*_comparison.py --replot <ts>` 默认写回 `results/<ts>_*.png`(overwrite);中央入口包一层 `copy2` 到 canonical PNG。
- `history_cli.accept` 必须把 trunk tip `copy2` 到 `canonical_output_path`——这是 draft / LaTeX 的稳定引用路径,trunk tip ≡ canonical PNG。
- `history_server` 仅 bind 127.0.0.1;`_safe_image_path` 校验路径必须在 `paper/figures/_history/` 子树防 `../` 越界;manifest 原子写(tempfile + `os.replace`)+ `lock_version` 自增防并发。
- `_history/` 整树 git-ignored;其内 `_rejected` 残留(如 fig4b/fig3c 的 smoke-test r1)**非 git 问题**,清理是 manifest+文件操作不是 git 操作。
- 评论内容寻址 `(before_sha, after_sha)`:trunk 前进**不自动** resolve / re-anchor 旧评论(故 fig2 评论仍 `open` 且 label 停在 v0→v1,虽 trunk 已到 v2)。

## 遗留进入 Phase 7 的状态(交接清单)

- **6 条 open 评论**:fig1 / fig2 / fig6b / fig8 / fig10b / fig_s1(全文见 Phase 7 计划与 `history_cli.py context-bundle <fig_id>`)。
- fig2 / fig3c v2 已 Accept 但**顶部 pane 标题缺陷未修**(共享根因)。
- `src/visualization/comparison.py` 未提交 diff = **混合**:保留 force-directed 标签 + `apply_paper_style(fig=fig)`(用户称赞);唯一需回退 = 删顶部标题那一 hunk(comparison.py:128-131)。
- `src/visualization/extra_sessions.py`(chance 线 gray→chance_red 重着色)+ `src/visualization/paper_style.py`(纯空白/注释整理)= 与 6 评论**无关的 drift**,用户决定**保留不动、排除出 comment-fix 提交、单列待其单独审查**。
- `results/*.png`(fig1/2/3c/6b canonical)被 2026-05-19/20 中央重生成覆盖,未提交。
