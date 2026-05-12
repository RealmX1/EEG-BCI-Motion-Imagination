# Stage 6: Process Summary — paper_draft_v3 → paper.pdf

**Date**: 2026-05-10
**Pipeline**: ARS academic-pipeline v3.7.0 (Stages 0–5 complete)
**Outcome**: Master's-thesis-scale paper successfully revised + finalized to publication-grade PDF
**Author**: 张博铭 (Bomin Zhang) — `davidzhangshs@gmail.com`
**Working tree**: `c:\Users\zhang\Desktop\github\EEG-BCI`

---

## §1 Executive Timeline

This pipeline ran over roughly one calendar day (2026-05-09 evening through 2026-05-10 21:56) and took a pre-existing v3 draft (`paper/drafts/paper_draft_v3.md`, 165,633 bytes / 917 lines) through the full ARS Stage 0 → Stage 5 sequence, producing a 36-page typeset PDF (`paper/build/paper.pdf`, 5,652,451 bytes), an integrated revised draft (`paper/drafts/paper_draft_v3.1.md`, 208,846 bytes / 1,010 lines, +43,213 bytes / +93 lines vs v3), a 35-concern Response-to-Reviewers Letter (`paper/reviews/response_to_reviewers_v3.1.md`, 67,661 bytes), and 17 Stage-3/4/4.5 audit-trail review documents. Including the late-cycle Phase 0 backfill and a Stage 4.5 surgical-fix pass, the pipeline executed 12 distinct stage transitions (0 → 1 → 2 → 2.5 → 3 → Phase 2.5 decision → 4 → Phase 0 backfill → 3' → 4' → 4.5 → 5).

The single most consequential inflection point was **mid-Stage-4 V4/V5 supersession**: the author externally completed surgical DAPT experiments V4 (single-dataset Stieger ablation) and V5 (channel-overlap ablation) on a parallel agent's compute, and the orchestrator inserted Step 1b (`paper/reviews/stage4_step1b_stat_recompute_v4v5.md`, 16,505 bytes) before continuing Stage 4 text revision. This single pivot upgraded §3.6 from a "weak negative result" framing into a **task-asymmetric mechanistic finding** by ruling out the Stieger-dominance and channel-heterogeneity confounds, leaving the MI-granularity-mismatch hypothesis as the surviving explanation. Without that supersession, the paper would have shipped a strictly weaker §3.6 narrative and Reviewer 3' would likely have escalated rather than returning Minor Revision.

---

## §2 Pipeline Stages Detail

### Stage 0 — RESEARCH (literature scoping, devil's advocate framing)
The `research_architect_agent` profiled the paper domain (EEG foundation models, motor imagery decoding, finger-level granularity), identified the 5-perspective reviewer panel (Editor-in-Chief / R1 Methodology / R2 Domain / R3 Cross-disciplinary / Devil's Advocate), and seeded the field-context document at `paper/reviews/stage3_phase0_field_analysis.md` (28,324 bytes). The Devil's Advocate frame in particular pre-loaded the HPO-budget-asymmetry concern that would surface as the Stage 3 CRITICAL.

### Stage 1 — OUTLINE / PLAN
The orchestrator entered Plan mode, synthesized the research output into a revision-and-typesetting roadmap, and obtained user approval before any writing. No artifact was produced beyond the in-conversation plan; this stage's role was purely gating user assent.

### Stage 2 — WRITE (input acquisition)
v3.md (`paper/drafts/paper_draft_v3.md`, 165,633 bytes / 917 lines) was a pre-existing draft authored by the user prior to ARS invocation. The pipeline therefore did not "generate" Stage 2 output; it ingested v3.md as the canonical input and preserved it byte-identical thereafter.

### Stage 2.5 — INTEGRITY (pre-review)
The integrity agent surfaced 4 issues in `paper/reviews/integrity_pre_review_v3.md` (33,302 bytes): (1) phantom author "H. Abuhijleh" in reference [8] (MEDIUM, removed); (2) Table 6 standard-deviation 13.82% vs Table 18 canonical 15.42% mismatch (MEDIUM, reconciled to 15.42%); (3) Table 11 XSI-FT cell mis-aligned to a non-canonical run (MINOR, aligned); (4) §3.1 subject list S05/S09 expansion to S04/S05/S09 (MINOR, expanded). The agent reported "PASS WITH NOTES"; per ARS spec, any MEDIUM mandates FAIL. The orchestrator surfaced this discrepancy honestly and the user chose **Option A** — apply the 4 fixes, preserve v3.md byte-identical, and write the corrected text to `paper/drafts/paper_draft_v3.0.1.md` (165,667 bytes).

### Stage 3 — PEER REVIEW (5-reviewer parallel)
Five reviewer subagents ran in parallel against v3.0.1.md, producing:
- `stage3_eic_review.md` (24,384 bytes)
- `stage3_r1_methodology_review.md` (31,155 bytes)
- `stage3_r2_domain_review.md` (31,866 bytes)
- `stage3_r3_perspective_review.md` (27,964 bytes)
- `stage3_devils_advocate_review.md` (42,994 bytes)
- `stage3_phase2_editorial_decision.md` (16,248 bytes) — meta-EIC consolidation
- `stage3_revision_roadmap.md` (27,354 bytes) — actionable plan

Total: **35 distinct concerns**, **1 CRITICAL** (Devil's Advocate #1.1: HPO budget asymmetry — CBraMod 51 trials × 11 dim vs EEGNet 32 trials × 7 dim creating an unfair comparison ceiling), **11 MAJOR**, balance MINOR/COSMETIC. Editorial verdict: **Major Revision**.

### Phase 2.5 — Orchestrator scope-tradeoff decision (user-driven)
Three user decisions defined the revision envelope:
1. **Scope**: chose **Option B** (skip P0.1 + P0.2 = ~140 GPU-hours of new experiments) over A (full re-experiments) or C (compromise);
2. **§3.7 framing**: chose **Option E** (reframe from "three-way decomposition" to "exploratory ablations") to avoid claiming statistical decomposition the design didn't support;
3. **§3.6 framing**: chose **Option A** (heavy reframe with task-asymmetric narrative) over B (status-quo defense) or C (light edit).

These combined into a "**B+E** + §3.6-A" stance. The CRITICAL was defended via the **(W) two-part stance**: Part A used Bergstra (2012) and Snoek (2012) effective-dimensionality calibration to argue CBraMod's 11-dim × 51 trials and EEGNet's 7-dim × 32 trials were equivalent in normalized search-space coverage (d^c metric); Part B reframed §3.7 itself, retiring the "decomposition" claim that the budget asymmetry would have undermined.

### Stage 4 — REVISION (3-subagent parallel dispatch + integrator)
The orchestrator partitioned the 35 concerns into three non-overlapping work-streams to minimize cross-section conflict:
- **Step 1** (`stage4_step1_stat_recompute.md`, 16,837 bytes) — recompute all stats from canonical run-tags; later superseded by:
- **Step 1b** (`stage4_step1b_stat_recompute_v4v5.md`, 16,505 bytes) — V4/V5 incorporation (see decision branch §3.4);
- **Step 2 Part A** (`stage4_step2_partA_dapt.md`, 56,219 bytes) — DAPT/§3.6 reframe block (Subagent A);
- **Step 2 Part B** (`stage4_step2_partB_capacity.md`, 53,448 bytes) — capacity / §3.7 / §2.5.1 (Subagent B);
- **Step 2 Part C** (`stage4_step2_partC_minor_lit_RR.md`, 50,830 bytes) — literature additions, MINOR fixes, R&R Letter seeds (Subagent C);
- **Step 2.5** (`stage4_step2.5_integration_report.md`, 11,821 bytes) — cross-section consistency audit (integrator);
- **Step 3** (`stage4_step3_figures_report.md`, 11,670 bytes) — 10 new figures generated.

Net: **28 surgical EDITs** + **16 new references [10]–[25]** + **10 figures** applied to v3.0.1.md → `paper/drafts/paper_draft_v3.1.md` (208,846 bytes / 1,010 lines, **+93 lines / +43,213 bytes** over v3.md).

### Phase 0 — P0.3 backfill (mid-cycle insertion)
The author externally completed the cross-subject CBraMod label-shuffle control on a parallel compute resource. Result: **pooled accuracy 49.58% vs the 90.68% headline = Δ −41.1 pp**, falling exactly within the pre-registered Scenario A acceptance band [48%, 52%]. The orchestrator inserted a backfill pass between Stage 4 completion and Stage 3' invocation, populating §3.9, R&R Letter §G.2 / §E.2 / Section H, and the abstract sentence. Without this insertion, Stage 3' would have flagged the placeholder as a residual issue.

### Stage 3' — RE-REVIEW (single consolidated synthesis agent)
A single synthesis agent — not the 5-reviewer panel — re-evaluated v3.1.md against the 35 Stage 3 concerns and the editorial roadmap. Output: `paper/reviews/stage3prime_rereview.md` (34,104 bytes). Verdict: **Minor Revision**. The CRITICAL (HPO budget asymmetry) was retired. Four cosmetic R-level fixes were issued: **R-1** §3.7 figure caption alignment; **R-2** §3.6.1 alternative-hypothesis explicit disclosure; **R-3** abstract P0.3 sentence; **R-4** terminology cleanup sweep (most importantly: legacy "~10K" parameter-count phrasing → corrected "~16K" for EEGNet).

### Stage 4' — RE-REVISE (light pass)
Per Stage 3' explicit recommendation, the orchestrator chose a **light** Stage 4' (apply R-1..R-4 inline) rather than a full re-revise cycle. The R-1..R-3 fixes landed cleanly. R-4 was implemented as targeted line edits (the brief named "lines ~465 + ~1398") rather than a full sweep — see decision branch §3.8.

### Stage 4.5 — FINAL INTEGRITY
A second integrity gate caught the R-4 sweep gap: **1 MEDIUM + 5 LOW** in `paper/reviews/stage4.5_final_integrity.md` (30,835 bytes). The MEDIUM was 5 surviving "~10K" residuals from incomplete R-4 sweep; 4 LOW were 3 uncited refs [11]/[17]/[22] plus minor cosmetic items. Surgical fixes applied → **PASS verdict** post-fix.

### Stage 5 — FINALIZE (LaTeX typeset)
The Markdown v3.1.md was converted to `paper/build/paper.tex` (72,135 bytes) with `references.bib` (9,047 bytes) using the article class + `xeCJK` (CJK support) + BibTeX. Compile clean: 0 errors, 7 cosmetic warnings (font substitution / overfull hbox margins). Output: `paper/build/paper.pdf` — **5,652,451 bytes / 36 pages** — and the Stage 5 build report at `paper/build/STAGE5_REPORT.md` (8,595 bytes). 21 figures were embedded.

---

## §3 Critical Decision Branch Points

### §3.1 Stage 2.5 spec interpretation discrepancy
- **Decision**: How to handle integrity agent's "PASS WITH NOTES" verdict when ARS spec mandates "ANY MEDIUM = FAIL".
- **Alternatives**: (a) accept agent verdict, proceed to Stage 3; (b) hard-fail per spec and re-run integrity; (c) surface discrepancy to user.
- **Path chosen**: (c) — surface honestly; user picked Option A (apply fixes, no full re-verify).
- **Why it won**: Preserved spec-conformance language for archival traceability while avoiding a redundant re-run on the same 4 issues. The user retained the authority over how strictly to apply spec.
- **Retrospective**: **Right call.** Auto-rubber-stamping the agent verdict would have left a future auditor unable to reconcile "PASS WITH NOTES" with the spec; surfacing it preserved the audit trail. The cost (one round-trip clarification) was minimal.

### §3.2 §3.6 reframe choice with concrete BEFORE/AFTER text
- **Decision**: Among three §3.6 framings (heavy reframe / status-quo defense / light edit), which best handles the DAPT negative result?
- **Alternatives**: (A) heavy task-asymmetric reframe; (B) defend original "DAPT helps in some configs" framing; (C) split the difference.
- **Path chosen**: (A) heavy reframe.
- **Why it won**: AI presented three options with concrete BEFORE/AFTER abstract-text snippets, making the trade-offs visible at the prose-quality level rather than abstract framing-level. User then picked (A) decisively.
- **Retrospective**: **Right call, and the presentation method generalizes.** The "make the option visible at the actual rendered text" pattern should be the default for any framing-tradeoff decision in future ARS runs. Cost: ~2× more prep work for AI vs single-recommendation. Benefit: drastically reduces back-and-forth.

### §3.3 HP dimension count correction (human catches AI estimation gap)
- **Decision**: How to defend the HPO-budget-asymmetry CRITICAL — what are the actual HP-space dimensions?
- **AI's first-pass estimate**: CBraMod ≈ 6–8 dim, EEGNet ≈ 3–4 dim.
- **User's intervention**: "your range may be wrong" — prompted re-read of `src/hpo/search_spaces.py`.
- **Corrected counts**: CBraMod **11-dim**, EEGNet **7-dim**.
- **Why this matters**: The Bergstra/Snoek d^c argument depends on accurate d. With wrong d, the (W) Part A defense would have been quantitatively unsound and Reviewer 3' might have called it out.
- **Retrospective**: **Human course-correction was essential.** AI should have read `src/hpo/search_spaces.py` first, not estimated. The lesson is: **whenever a numeric defense argument depends on a code-readable quantity, read the code first.**

### §3.4 V4/V5 supersedes Step 1 mid-pipeline
- **Decision**: Author notified mid-Stage-4 that V4/V5 surgical experiments completed externally on a parallel agent's compute. Continue with stale Step 1 stats or pause and integrate?
- **Alternatives**: (a) finish Stage 4 with Step 1 stats, append V4/V5 in a Stage 4'' addendum; (b) pause, dispatch Step 1b amendment, integrate before Stage 2 text revision; (c) defer V4/V5 to a v3.2 future revision.
- **Path chosen**: (b) Step 1b dispatch.
- **Why it won**: V4/V5 fundamentally changed the §3.6 finding from "weak negative result" to "task-asymmetric mechanistic finding". Stale stats would have produced text that needed full re-revise anyway, costing more total work.
- **Retrospective**: **Right call, and the highest-leverage decision in the pipeline.** Without (b), §3.6 would have shipped strictly weaker. Future ARS runs should treat external evidence arrival as a first-class pipeline event.

### §3.5 Master finding upgrade (§3.6) — emergent from V4/V5
- **Decision**: How to frame §3.6 once V4/V5 ruled out two of three confounds?
- **Confounds at issue**: (i) Stieger-dataset dominance (V4 ablated) ; (ii) channel-heterogeneity (V5 ablated); (iii) MI-granularity mismatch (only surviving hypothesis).
- **Path chosen**: Promote §3.6 from "DAPT didn't help" to "task-asymmetric finding: DAPT lifts coarse-grained MI but fails on finger-level due to granularity mismatch".
- **Why it won**: The mechanistic narrative is publishable on its own; the original "negative result" framing was master's-thesis-acceptable but reviewer-fragile.
- **Retrospective**: **Right call. The author's external compute investment shaped what AI could write — a real-world demonstration that compute resourcing is a paper-quality lever, not just an experimental lever.**

### §3.6 P0.3 mid-Stage-3' notification
- **Decision**: Author notified mid-Stage-3' invocation that P0.3 (cross-subject label-shuffle control) had completed externally. Hold Stage 3' until P0.3 backfilled, or backfill after?
- **Path chosen**: Insert "Phase 0 backfill" pass between Stage 3' completion and Stage 4' start, so Stage 3' reviewed v3.1.md *with* P0.3 already integrated.
- **Why it won**: A reviewer seeing a P0.3 placeholder would have flagged it as residual issue, potentially escalating verdict from Minor to Major Revision.
- **Retrospective**: **Right call.** Pipeline order was adapted to external evidence arrival. Reinforces §3.4's lesson that external evidence is a first-class event.

### §3.7 Stage 3' verdict triage (light Stage 4' vs full Stage 4'')
- **Decision**: Stage 3' returned Minor Revision with 4 R-level cosmetic fixes. Full Stage 4 re-revise cycle (5-reviewer / 3-subagent dispatch / integrator) or light inline pass?
- **Path chosen**: Light inline pass per Stage 3' explicit recommendation.
- **Why it won**: A full Stage 4'' for cosmetic fixes would have been ~10× the cost for ≤5% benefit. Stage 3' itself recommended light pass.
- **Retrospective**: **Right call on cost-benefit, but exposed the R-4 sweep gap (see §3.8).** The lesson: when going light, the brief specification matters more, not less.

### §3.8 Stage 4.5 catch — R-4 sweep incompleteness (AI propagation drift)
- **Decision**: Stage 4' implementer received the R-4 brief naming "lines ~465 + ~1398" and treated those as exhaustive. Stage 4.5 audit caught **5 surviving "~10K" residuals** + 3 uncited refs.
- **Root cause**: Brief specificity ("lines ~465 + ~1398") was read as complete enumeration; audit's intent was "full sweep, these are example anchors". This is **propagation drift between human brief and AI execution**.
- **Path chosen**: Stage 4.5 audit (`paper/reviews/stage4.5_final_integrity.md`) caught it; surgical fix pass applied; **PASS** post-fix.
- **Retrospective**: **AI failure mode that the integrity gate caught.** The lesson is double-edged: (a) future briefs should explicitly say "OR similar elsewhere — full sweep" when intent is exhaustive; (b) independent integrity gates *do* catch this class of error reliably and are worth the audit cost. Also see §6.

---

## §4 Human-AI Collaboration Patterns

### §4.1 Where AI provided value the human couldn't replicate at scale
- **5-reviewer parallel review**: Stage 3 dispatched EIC + R1 + R2 + R3 + Devil's Advocate as concurrent subagents, generating 5 distinct review documents (24K–43K bytes each) within a single wall-clock block. A human running this serially would take 2–5 days; AI did it in roughly 30 minutes of orchestrator wall-clock.
- **3-subagent revision partition**: Stage 4 Step 2 split into Parts A/B/C running in parallel, each producing 50K+ bytes of revised text. The integrator (Step 2.5) then audited cross-section consistency. Human equivalent: a research group of 3 + 1 editor.
- **Integrity gates (Stage 2.5 + Stage 4.5)**: Mechanical cross-checks of statistical claims against canonical run-tags, reference list against citations, table-cell alignment. AI executes these systematically; humans skip steps under fatigue.

### §4.2 Where the human caught errors AI missed
- **HP dimension count** (§3.3): User challenged AI's 6-8 / 3-4 dim estimate; correct values 11 / 7 came from the code. AI should have grep'd `src/hpo/search_spaces.py` first.
- **§3.6 reframe authority**: AI presented options with concrete text but did *not* pick. The reframe direction (heavy / status-quo / light) is a paper-voice decision and was correctly deferred to user.
- **External-evidence injection** (V4/V5, P0.3): User provided experimental results AI cannot generate. AI's role was integration, not generation.

### §4.3 Genuinely human-authority decisions
- **Scope tradeoffs**: skip P0.1 + P0.2 (~140 GPU-h) vs run them. Budget allocation is a master's-thesis economic decision, not an AI-judgment one.
- **Framing voice**: §3.6 heavy reframe vs status-quo. Authorial voice belongs to the author.
- **Spec interpretation latitude**: Stage 2.5 PASS-WITH-NOTES vs strict-FAIL. AI surfaced; user decided.
- **Baseline-replacement authority** (per project CLAUDE.md): "Baseline 替换必须由开发者明确提出" — the project guidelines explicitly deny AI baseline-replacement authority. Pipeline respected this throughout.

### §4.4 Work AI executed reliably without supervision
- **Surgical EDIT application**: 28 EDITs in Stage 4 + 4 R-level fixes in Stage 4' + 6 fixes in Stage 4.5. Each is a string-match-and-replace task. AI does this without error when the brief is precise.
- **Reference list maintenance**: Adding [10]–[25] with consistent citation format, ordering, and bib entries. Mechanical when scope is bounded.
- **Cross-document consistency audits**: Stage 4 Step 2.5 integration report systematically checked table-figure-text agreement across 1,010 lines. This is the kind of work humans skip under deadline.
- **LaTeX typesetting** (Stage 5): Markdown → `paper.tex` + BibTeX → 36-page PDF with 21 figures, 0 errors, 7 cosmetic warnings. Mechanical translation under known templates.

### §4.5 Mixed-authority decisions (collaboration sweet spot)
- **(W) two-part HPO defense**: AI proposed the structure; user corrected the dim counts; AI completed the d^c calibration math. Neither party alone could have produced this defense — AI brought the Bergstra/Snoek literature framing; user brought the codebase ground truth.
- **§3.7 Option E reframe**: AI flagged "decomposition" as overclaim; user agreed and chose E; AI executed the reframe text. Joint epistemic work.

---

## §5 AI Self-Reflection — What Went Well

### §5.1 Honest spec-interpretation surfacing (Stage 2.5)
The integrity agent's "PASS WITH NOTES" verdict conflicted with the ARS spec ("ANY MEDIUM = FAIL"). AI could have rubber-stamped the agent. Instead AI surfaced the discrepancy ("the agent says PASS but spec says FAIL — how do you want to handle?") and let the user decide. **This preserved audit trail integrity.** The cost was one user round-trip; the benefit is a future auditor can reconstruct the decision.

### §5.2 Adaptive pipeline ordering for external evidence (V4/V5 + P0.3)
Twice — V4/V5 mid-Stage 4 and P0.3 mid-Stage 3' — external experimental evidence arrived asynchronously. AI did not insist on the canonical pipeline order. Instead AI inserted Step 1b (V4/V5) and Phase 0 backfill (P0.3) precisely where they would maximize downstream quality. **This treated the pipeline as a quality function to optimize, not a script to execute.** Future ARS runs should formalize "external evidence arrival" as a first-class event with insertion-point heuristics.

### §5.3 Concrete option presentation for framing decisions (§3.6 reframe)
For the §3.6 heavy/status-quo/light tradeoff, AI presented three concrete BEFORE/AFTER abstract-text snippets rather than abstract framing descriptions. The user picked decisively. **The lesson: when the decision is about prose quality, render the prose, don't describe the framing.** This pattern should generalize to any §-level reframe decision.

### §5.4 Multi-touch integration audit (Stage 4 Step 2.5)
After Subagents A/B/C produced their independent revisions, Step 2.5 (integrator) systematically audited cross-section consistency: abstract claims vs §3.6 body, §1.4 contributions list vs §7 conclusions, Table 6 SD vs Table 18 canonical. **This caught at least 3 cross-section drift items before they reached Stage 3' review.** The 4-subagent pattern (3 worker + 1 integrator) is recommended for any multi-section revision.

### §5.5 Independent integrity gates catching propagation drift (Stage 4.5)
Stage 4.5 was not strictly required by the ARS roadmap once Stage 3' returned Minor Revision and Stage 4' was a light pass. Running it anyway caught the R-4 sweep gap — 5 surviving "~10K" residuals that would have shipped to PDF. **The lesson: post-light-revise integrity gate is high-leverage.** The cost was modest (~30K-byte audit doc); the saved cost was a post-publication erratum.

---

## §6 AI Self-Reflection — What Could Be Better

### §6.1 R-4 sweep incompleteness (propagation drift)
The R-4 brief named "lines ~465 + ~1398" as anchors. AI as Stage 4' implementer treated these as **exhaustive enumeration** rather than **example anchors with full-sweep intent**. Five "~10K" residuals slipped through and were caught only by Stage 4.5. **Root cause: ambiguity in the brief between "fix these" and "fix these and similar elsewhere".** Lesson:

- For implementer subagents, default reading of "lines X + Y" should be exhaustive.
- For brief-writers (orchestrator), when intent is "sweep", say "**OR similar elsewhere — full sweep**" explicitly.
- For the next ARS pipeline version: add an explicit `sweep: true|false` flag to R-level fix items.

### §6.2 HP dimension count initial guess (lazy estimation)
AI's first-pass estimate (6–8 / 3–4 dim) for the HPO-budget defense was imprecise enough that the user caught it. The correct values (11 / 7) were one `grep -n` away in `src/hpo/search_spaces.py`. **Lesson: whenever a numeric defense depends on a code-readable quantity, read the code first, estimate never.** This is a baseline AI hygiene issue, not a pipeline-design issue.

### §6.3 Pipeline coupling between Stage 3' and Stage 4'
Stage 3' explicitly recommended "light Stage 4'". AI followed this recommendation — but the Stage 3' brief generation and Stage 4' brief generation were sequential rather than co-designed. If Stage 3' had been written with the Stage 4' implementer's reading style in mind, the R-4 sweep ambiguity could have been avoided. **Lesson: when one stage writes a brief consumed by the next stage's subagent, the first stage should anticipate the implementer's reading.**

### §6.4 Stage 2.5 spec definition latitude
ARS spec says "ANY MEDIUM = FAIL" but the integrity agent reported "PASS WITH NOTES". This indicates the agent's prompt is not tightly synchronized with the spec language. **Lesson: integrity-agent prompts should quote the spec verbatim for the verdict-rule, not paraphrase.** Future ARS versions should treat this as a prompt-engineering bug.

### §6.5 Insufficient pre-pipeline file-state snapshotting
The Phase 0 P0.3 backfill required reconstructing what v3.1.md looked like *before* Phase 0 to know what to insert. AI could do this from git diff, but a snapshot would have been faster. **Lesson: at major stage transitions, snapshot the working draft (not just the start-of-pipeline draft).** Stage 4.5 worked from `paper_draft_v3.1.md` post-Stage-4'-light, but if a later issue had required reverting Stage 4' specifically, recovery would have been harder.

---

## §7 Recommendations for Future ARS Runs

### §7.1 R-level brief sweep semantics
For Stage 4' (re-revise) cleanup briefs of the R-4 type ("terminology cleanup"): when listing "lines X + Y" as anchors, **always include explicit "OR similar elsewhere — full sweep" instruction** when intent is exhaustive. Better yet, add a structured `scope: anchored | sweep` field to each R-level item. The Stage 4.5 catch (§3.8) is the canonical case for this rule.

### §7.2 External-evidence arrival as first-class event
Maintain a "deferred backfill checklist" in the orchestrator state. When the user notifies external evidence (V4/V5 mid-Stage-4, P0.3 mid-Stage-3'), the orchestrator should:
1. Acknowledge receipt with a snapshot of current pipeline position.
2. Decide insertion point (before next stage that consumes the evidence vs after current stage completes).
3. Document the pivot in the eventual Stage 6 process summary.

### §7.3 4-subagent pattern for multi-section revisions
For any Stage 4 revision touching ≥3 sections of the manuscript, dispatch 3 worker subagents partitioned by section + 1 integrator subagent for cross-section audit. The Stage 4 Step 2 (Parts A/B/C) + Step 2.5 (integrator) implementation is the proven template. **Do not rely on a single Stage-4 subagent for multi-section work** — cross-section drift is undetectable without an explicit integrator pass.

### §7.4 Concrete-text option presentation for framing decisions
For any §-level reframe decision (status-quo / light / heavy), present 2–3 concrete BEFORE/AFTER abstract-text snippets. The user makes the decision once, with full visibility, rather than three rounds of "what would option A actually look like in the abstract?". The §3.6 decision (§3.2) is the canonical case.

### §7.5 Stage 4.5 (post-light-revise integrity gate) as default
When Stage 3' returns Minor Revision and Stage 4' is light, **default to running Stage 4.5 anyway**. The cost is modest (one audit subagent + 30K-byte report); the benefit is catching propagation drift that would otherwise reach the PDF. The Stage 4.5 catch in this run paid for itself immediately.

### §7.6 Code-grounded numeric defenses
Whenever a defense argument depends on a code-readable quantity (HP-space dimension count, parameter count, dataset size, GPU-hour budget), the orchestrator should grep the code **before** drafting the defense, not after the user catches a guess. Adding a "code-anchored claims pre-check" to Stage 4 brief generation would catch this class of error.

---

## §8 Final Reflections

### §8.1 The single biggest factor that shaped outcome quality
**External evidence arrival, asynchronous to the pipeline, treated as first-class event.** V4/V5 (§3.4) upgraded §3.6 from weak-negative to task-asymmetric mechanistic finding. P0.3 (§3.6) closed the Devil's Advocate residual concern about cross-subject CBraMod's "too good" headline. Both arrived mid-pipeline, both required adaptive insertion, both fundamentally improved the paper. The pipeline's willingness to pause, integrate, and continue was the high-leverage behavior. A rigid "complete Stage N before considering external input" pipeline would have shipped strictly weaker work.

### §8.2 The single biggest gap between planned and achieved
**The R-4 sweep gap.** Stage 3' issued R-4 as a clean "terminology cleanup". Stage 4' implementer interpreted the line-anchors as exhaustive. Stage 4.5 caught 5 residuals + 3 uncited refs. The plan was "light cleanup + final integrity"; the achievement required two integrity-gate iterations. This is a 5–10% slippage in execution-cost terms, but the more important slippage is conceptual: the pipeline assumed "light Stage 4' = trivial Stage 4'". It is not. Light briefs require **more** specification rigor, not less, because the implementer has less context to disambiguate from.

### §8.3 Publishability as master's thesis
**Yes.** The paper at `paper/build/paper.pdf` (36 pages, 5.4 MB, 21 figures, 25 references) presents three publishable contributions: (1) a 21-subject within / cross / XSI-FT comparison of CBraMod vs EEGNet at finger-level granularity demonstrating that foundation-model advantage holds in cross-subject pooling but degrades in within-subject finger-level decoding; (2) a channel-reduction ablation across 4/8/32/61/128 channels with empirical scaling-curve characterization; (3) a task-asymmetric DAPT finding ruling out two confounds and isolating the MI-granularity-mismatch hypothesis as the surviving explanation. The Devil's-Advocate CRITICAL was retired; Stage 3' returned Minor Revision; Stage 4.5 returned PASS post-fix; LaTeX compile is clean. **It is master's-thesis-ready and would survive review at a workshop or short-paper venue. A full-conference submission would benefit from the deferred P0.1/P0.2 experiments (the ~140 GPU-h block the user chose to skip).**

---

## §9 Artifact Inventory

### Drafts
| Path | Bytes | Lines | Notes |
|---|---:|---:|---|
| `paper/drafts/paper_draft_v3.md` | 165,633 | 917 | Original input (Stage 0/1/2). Preserved byte-identical. |
| `paper/drafts/paper_draft_v3.0.1.md` | 165,667 | — | Stage 2.5 integrity-fixed (4 fixes). |
| `paper/drafts/paper_draft_v3.1.md` | 208,846 | 1,010 | Stage 4 + Phase 0 + Stage 4' + Stage 4.5 final. |

### Review documents (Stage 2.5 → Stage 4.5)
| Path | Bytes | Stage |
|---|---:|---|
| `paper/reviews/integrity_pre_review_v3.md` | 33,302 | 2.5 |
| `paper/reviews/stage3_phase0_field_analysis.md` | 28,324 | 3 (research) |
| `paper/reviews/stage3_eic_review.md` | 24,384 | 3 |
| `paper/reviews/stage3_r1_methodology_review.md` | 31,155 | 3 |
| `paper/reviews/stage3_r2_domain_review.md` | 31,866 | 3 |
| `paper/reviews/stage3_r3_perspective_review.md` | 27,964 | 3 |
| `paper/reviews/stage3_devils_advocate_review.md` | 42,994 | 3 |
| `paper/reviews/stage3_phase2_editorial_decision.md` | 16,248 | 3 |
| `paper/reviews/stage3_revision_roadmap.md` | 27,354 | 3 |
| `paper/reviews/stage4_step1_stat_recompute.md` | 16,837 | 4 |
| `paper/reviews/stage4_step1b_stat_recompute_v4v5.md` | 16,505 | 4 (V4/V5) |
| `paper/reviews/stage4_step2_partA_dapt.md` | 56,219 | 4 |
| `paper/reviews/stage4_step2_partB_capacity.md` | 53,448 | 4 |
| `paper/reviews/stage4_step2_partC_minor_lit_RR.md` | 50,830 | 4 |
| `paper/reviews/stage4_step2.5_integration_report.md` | 11,821 | 4 |
| `paper/reviews/stage4_step3_figures_report.md` | 11,670 | 4 |
| `paper/reviews/stage3prime_rereview.md` | 34,104 | 3' |
| `paper/reviews/stage4.5_final_integrity.md` | 30,835 | 4.5 |
| `paper/reviews/response_to_reviewers_v3.1.md` | 67,661 | R&R Letter |
| `paper/reviews/stat_recompute_runner.py` | 34,980 | Step 1 helper |
| `paper/reviews/stat_recompute_v4v5_runner.py` | 31,314 | Step 1b helper |

### Build artifacts (Stage 5)
| Path | Bytes | Notes |
|---|---:|---|
| `paper/build/paper.pdf` | 5,652,451 | **Final 36-page PDF** |
| `paper/build/paper.tex` | 72,135 | LaTeX source (article + xeCJK) |
| `paper/build/references.bib` | 9,047 | 25 refs |
| `paper/build/paper.aux` | 26,957 | LaTeX aux |
| `paper/build/paper.bbl` | 5,376 | BibTeX output |
| `paper/build/paper.log` | 38,611 | Compile log (0 errors, 7 cosmetic warnings) |
| `paper/build/paper.toc` | 5,555 | Table of contents |
| `paper/build/STAGE5_REPORT.md` | 8,595 | Stage 5 build report |

### Figures (21 unique, embedded into PDF)
Located in `paper/build/figures/` (and mirror copies in `paper/figures/`):
`32ch_comparison.png`, `channel_method_ranking_flip.png`, `channel_scaling_curve.png`, `cross_subject_pooling_forest.png`, `dapt_v1_v5_forest.png`, `exploratory_ablation_overview.png`, `extra_sessions_binary.png`, `extra_sessions_paradigm_binary.png`, `extra_sessions_strategy_comparison.png`, `extra_sessions_ternary.png`, `fig01_within_128_binary.png`, `fig02_cross_128_binary.png`, `fig03b_32ch_fdr_cross.png`, `fig06_xsift_binary.png`, `fig06b_xsift_ternary.png`, `fig5_4ch_optimal_vs_neg_control.png`, `further_pretraining.png`, `inference_latency.png`, `sensitivity_scaling.png`, `subject_heatmap.png`, plus `extra_sessions_binary_copy.png` (legacy duplicate).

### This document
| Path | Stage |
|---|---|
| `paper/reviews/stage6_process_summary.md` | 6 (this file) |

---

**End of Stage 6 Process Summary. Pipeline ARS academic-pipeline v3.7.0 closed at 2026-05-10.**
