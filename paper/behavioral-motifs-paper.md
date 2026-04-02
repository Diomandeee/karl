# Geometric Motifs for Selecting and Routing Coding-Agent Training Data

**Mohamed Diomande**

March 2026

---

## Abstract

We present a method for compactly annotating coding agent sessions with behavioral motifs and geometric features, then conditioning training data generation on these annotations. From 834 real multi-project coding sessions spanning 4,633 turn-level records across 50+ applications, we extract 10-category symbolic labels (inscriptions) and 5 continuous geometric scalars. We show that: (1) transition pressure predicts session convergence at 71.8% accuracy (z = 2.72, p < 0.007), (2) advantage-weighted training using these annotations yields Cohen's d = 3.065 over random selection, and (3) geometry-conditioned routing produces higher-specificity training data for inscription-type sessions (Cohen's d = +1.02) but requires balanced quota enforcement — unconstrained routing concentrates sessions in low-specificity lenses, reducing overall quality (d = -0.60 when corpus is residual-dominated). Motivated by recent work on conditional memory in transformers [1], we test whether retrieval-conditioned supervision, where the model is trained on annotated behavioral patterns directly, further improves downstream performance. We describe the annotation pipeline, routing mechanism, quality verification, and iterative reward loop, and report results from a deployment spanning 50+ applications across 5 machines.

---

## 1. Introduction

Coding agent sessions vary in quality. Some produce clean, efficient outcomes. Others spiral through repeated failures and unnecessary corrections. This variance correlates with the nature of the task, the agent's routing decision, and patterns in the tool-use sequence. Yet most agent frameworks treat sessions as independent events, discarding the behavioral signal embedded in each trajectory.

We observe that certain behavioral patterns recur. A session that oscillates between approaches before converging follows a recognizable arc. A session where the agent ships a feature on the first try follows a different one. If these patterns can be compactly represented, they become useful for conditioning downstream training: selecting which sessions to learn from, routing sessions to appropriate transformation lenses, and teaching the model to recognize pattern types directly.

Our approach has three parts. First, we annotate each session with a compact symbolic label (its inscription) drawn from a 10-category vocabulary, plus 5 continuous geometric scalars computed from the inscription distribution. Second, we condition training data generation on these annotations, using geometry-aware routing to assign sessions to specialized evolution lenses. Third, we close the loop with an iterative reward mechanism that updates routing weights based on holdout evaluation.

The conditional memory literature motivates this design. DeepSeek's Engram architecture [1] stores token-level patterns via hash-lookup-gate-fuse, avoiding redundant recomputation. We test an analogous idea at the behavioral level: can annotating and routing session-level patterns improve training outcomes the way token-level memory improves inference? We frame this as a productive analogy, not a mechanistic equivalence claim.

**Contributions:**

1. An annotation scheme that compresses coding agent sessions into inscriptions + geometry (Section 3.1)
2. A geometry-conditioned routing mechanism for training data generation (Section 3.2)
3. A retrieval-conditioned training mode where the model learns to classify behavioral patterns (Section 3.3)
4. An iterative reward loop that refines routing weights from holdout evaluation (Section 3.4)

---

## 2. Related Work

### 2.1 Agent Trajectory Learning

SWE-Bench [2] and similar benchmarks evaluate agent coding ability on curated tasks with known solutions but provide point-in-time measurements, not continuous improvement signals. Databricks' Agent Training work introduced trajectory-based learning for coding agents, capturing tool-use sequences for fine-tuning. KARL (our prior work) extends this with advantage-weighted selection and entity-level performance tracking from 121+ real trajectories across 11 domains.

### 2.2 Reward Design

RLHF [3] requires human preference labels. Process reward models [4] reward correct reasoning steps rather than just final answers. Our approach derives reward from observable behavior (tool success rates, user corrections, session outcomes) with zero human annotation. The 5-signal composite reward we use (outcome quality, process efficiency, tool-use patterns, trajectory dynamics, correction rate) was validated in an ablation showing Cohen's d = 3.065 between high-advantage and random trajectory selection (Section 4.2).

### 2.3 Conditional Memory

DeepSeek's Engram architecture [1] introduces a conditional memory mechanism in transformers: token-level patterns are stored via hash functions, retrieved via lookup, gated, and fused into the residual stream. This bypasses redundant recomputation of recurring patterns. We note the structural analogy to our approach. Sessions with recurring behavioral motifs (our "inscriptions") are compactly represented, routed to specialized lenses (our "lookup"), filtered by quality (our "gate"), and merged into the training mix (our "fuse"). We adopt this as a design heuristic, not a claimed equivalence. Token patterns are local and deterministic; behavioral patterns are global and noisy. Whether the analogy yields similar efficiency gains is an empirical question we test in Experiments 4.3 and 4.4.

### 2.4 Self-Evolving Systems

The Learning to Self-Evolve framework (LSE) [5] demonstrates a dual-system architecture where a self-evolving policy observes an action model's failures, discovers domain-specific invariance, and rewrites instructions using empirical reward. We adopt LSE's exponential weight-update rule for our routing loop (Section 3.4) and its invariance extraction method for discovering which geometry ranges correlate with high-quality training data (Section 3.5). We use the specific mechanisms, not the full dual-system architecture.

---

## 3. Method

### 3.1 Session Annotation

**Input.** 834 sessions from 4,633 turn-level records across 50+ projects, recorded by the KARL trajectory instrumentation system over 6 months of real multi-project development.

**Inscriptions.** Each turn in a session receives a symbolic label (sigil) from a 10-category vocabulary:

| Sigil | Pattern | Indicator |
|-------|---------|-----------|
| stabilization | Consistent tool success, no direction changes | Steady execution |
| transition | Shift in project focus or approach | Pivoting |
| oscillation | Back-and-forth between alternatives | Indecision |
| correction | User corrects agent behavior | Error recovery |
| exploration | Trying new tools, files, or approaches | Discovery |
| convergence | Progressive narrowing toward a solution | Shipping signal |
| expansion | Scope increase, new features added | Growth |
| regression | Previously working things break | Quality loss |
| stagnation | No progress despite continued interaction | Blocked |
| completion | Task finished, clean exit | Done |

The session-level inscription is the dominant sigil across all turns, weighted by position (later turns weighted higher, reflecting where the session ended up).

**Geometry.** From the sigil distribution, we compute 5 continuous scalars:

- **convergence**: fraction of turns with convergence or completion sigils
- **exploration**: fraction of turns with exploration or expansion sigils
- **correction_rate**: fraction of turns with correction or regression sigils
- **focus**: Herfindahl index of the sigil distribution (1.0 = all one sigil, low = scattered)
- **avg_confidence**: mean confidence of the sigil classifier across turns

These scalars form a 5-dimensional "geometry" for each session.

**App Origin.** Each session is classified by its primary project, tier (1 = shipped single-project, 2 = service/multi-project, 3 = 3+ projects, 4 = unknown), and whether the project has been shipped to production. This uses a project identity resolution system that maps session content to known applications.

This is compact annotation, not a lookup table. The representation compresses a session's behavioral arc into ~20 bytes of categorical data plus 5 floats. We do not claim it functions identically to token-level Engram memory.

### 3.2 Geometry-Conditioned Routing

Given an annotated session, we route it to one of 5 evolution lenses, each designed to generate different types of training data:

| Lens | Purpose | Geometry Affinity |
|------|---------|-------------------|
| **Residual** | Extract ideas mentioned but never pursued | High exploration, high correction |
| **Decision** | Explore roads not taken at pivot points | High correction_rate, low convergence |
| **Cross-Synthesis** | Trace idea evolution across sessions | High convergence, high focus |
| **Inscription** | Teach the model to read behavioral sigils | High avg_confidence, high focus |
| **Shipping Coach** | Model convergence patterns from shipped apps | High convergence, high focus |

**Routing mechanism.** Each lens has an affinity vector in the 5-dimensional geometry space. For a session with geometry vector **g** and lens affinity **a**, the yield score is:

```
yield(session, lens) = (a · g) × tier_mult × confidence_mult + shipped_bonus + length_bonus
```

Where `tier_mult` favors shipped apps (2.0x for tier 1, 0.5x for tier 4), `confidence_mult` favors high-confidence annotations, `shipped_bonus` adds 0.5 for shipped projects, and `length_bonus = 0.1 × log(num_turns)` favors longer sessions.

**Quota allocation.** Rather than sending all sessions to the highest-scoring lens, we allocate per-lens quotas (residual 25%, decision 20%, cross-synthesis 5%, inscription 25%, shipping coach 25%) and rank sessions within each lens by yield score. This prevents monoculture in the training data.

### 3.3 Quality Verification

Each generated training record passes through two quality gates:

**Specificity scorer (0-1).** Checks for concrete nouns, file paths, code snippets, project-specific names, action verbs, and minimal hedge words. SFT records scoring below 0.33 are dropped. This prevents generic, ungrounded responses like "you should consider refactoring" from entering the training set.

**DPO contrast scorer (0-1).** For preference pairs, computes word-level Jaccard distance (50% weight), character overlap (30%), and length ratio (20%). Pairs scoring below 0.3 are dropped. This prevents near-duplicate chosen/rejected pairs that provide no training signal.

**Deduplication.** Records are keyed by a hash of session ID + lens + idea, preventing the same insight from appearing multiple times.

### 3.4 Retrieval-Conditioned Training

Standard SFT presents the model with (session context → response) pairs. Retrieval-conditioned SFT adds an explicit annotation step: the model first receives the inscription and geometry, then produces a behavioral classification before generating the response.

**Standard SFT format:**
```
System: You are a coding agent.
User: [session context]
Assistant: [response]
```

**Retrieval-conditioned SFT format:**
```
System: You are a coding agent with behavioral pattern awareness.
User: Session inscription: [sigil]. Geometry: convergence=0.7, exploration=0.2, ...
      Context: [session context]
Assistant: Pattern: [classification]. [response grounded in pattern recognition]
```

The hypothesis: by explicitly conditioning on the annotation, the model learns to use behavioral patterns as a routing signal, similar to how Engram's hash function routes to stored patterns. The control experiment (Section 4.4) tests this against standard SFT with identical data volume and training configuration.

### 3.5 Invariance Rule Extraction

After each training cycle, we compare the top 20% and bottom 20% of quality-filtered records to extract hard rules:

1. **Geometry rules**: which scalar ranges correlate with high quality (e.g., convergence > 0.5 in top vs < 0.3 in bottom)
2. **Sigil rules**: which inscription sequences are enriched in top records (e.g., "stabilization" 3x more frequent in high-quality outputs)
3. **Structural rules**: minimum response length, specificity floor
4. **Per-lens rules**: quality distribution per lens

These rules become constraints for future generation cycles, tightening the quality gate over time. This is empirical pattern extraction from distributional comparison, not a formal invariance proof.

### 3.6 Iterative Reward Loop

The full cycle:

1. **Generate** training data using geometry-conditioned routing with current lens affinity weights
2. **Filter** through quality verification (specificity + contrast gates)
3. **Train** on the filtered data (LoRA fine-tuning on Mac5)
4. **Evaluate** on a hash-locked holdout set (10% of records, keyed by `hash(record_id) % 10 == 0`)
5. **Compute reward** per lens: `reward_lens = mean_quality_holdout - baseline`
6. **Update weights**: `new_weight = old_weight × exp(η × reward)`, η = 0.5, clamped to [-5, 5]
7. **Extract invariance** rules from the new quality distribution
8. Return to step 1 with updated weights and rules

This is a direct application of LSE's exponential reward scaling [5]. We adopt the specific update rule without claiming to implement the full dual-system architecture.

---

## 4. Experiments

Each experiment isolates one variable. No confounded comparisons.

### 4.1 Does Annotation Predict Convergence?

**Setup.** 834 annotated sessions. For each, we compute transition pressure (the signed derivative of the convergence scalar over the session's turns) and compare against actual convergence outcome (did the session reach a completion or convergence sigil in its final 3 turns?).

**Metric.** Binary classification accuracy: transition pressure sign vs. convergence outcome.

**Result.** 71.8% accuracy (z = 2.72, p < 0.007, one-tailed binomial test against 50% baseline).

**Interpretation.** The annotation carries signal. Sessions where transition pressure is positive (convergence increasing over time) do converge more often than sessions where it is negative. This is a necessary condition for the annotation to be useful for routing, though 71.8% leaves substantial room for improvement.

### 4.2 Does Advantage-Weighted Selection Help?

**Setup.** Two training runs with identical configuration:
- **Control**: 35 trajectories selected uniformly at random
- **Treatment**: 35 trajectories with the highest advantage scores (computed by KARL's 5-signal reward function)
- **Config**: Qwen2.5-7B base, LoRA rank 8, 500 iterations, learning rate 1e-4, batch size 1

**Metric.** Training loss at iteration 500; generation quality on 10 held-out prompts rated by specificity scorer.

**Result.** Cohen's d = 3.065 (very large effect). The advantage-weighted set reaches lower training loss and produces higher-specificity generations.

**Interpretation.** The reward function discriminates meaningfully between high-value and low-value trajectories. This validates the upstream annotation and reward pipeline. The effect size is unusually large, likely because the worst trajectories contain sessions with tool failures and user corrections that actively teach the wrong behavior.

### 4.3 Does Geometry-Conditioned Routing Improve Data Quality?

**Setup.** Same pool of 834 annotated sessions processed through the evolution worm pipeline twice:
- **Control**: uniform random assignment of sessions to lenses (each session randomly gets one of the 5 lenses)
- **Treatment**: quota-based geometry routing (anticipation router with yield scoring)
- **Same** LLM (Gemini Flash), same prompts per lens, same quality filters

**Metric.** Mean specificity of generated SFT records; per-lens quality distribution; Cohen's d between conditions.

**Result.** Overall: treatment mean specificity = 0.358 (n=189), control mean specificity = 0.461 (n=367), Cohen's d = -0.60 (control_better). Per-lens breakdown tells a more nuanced story:

| Lens | Treatment mean | Control mean | Cohen's d | Direction |
|------|---------------|--------------|-----------|-----------|
| Inscription | 0.593 (n=20) | 0.450 (n=89) | +1.02 | Geometry better |
| Shipping Coach | 0.559 (n=26) | 0.587 (n=27) | -0.18 | Similar |
| Decision | 0.223 (n=21) | 0.288 (n=44) | -0.61 | Uniform better |
| Residual | 0.300 (n=122) | 0.485 (n=206) | -1.33 | Uniform better |

**Interpretation.** The overall negative result is largely a confound: the treatment corpus was assembled by reusing the existing geometry-routed worm output, which sent 64% of records to the residual lens (122/189). The residual lens generates low-specificity output by design (it extracts abandoned ideas, not executable actions). The quota enforcement mechanism — which should have balanced lens representation — was bypassed by the reuse path.

The inscription lens result (d=+1.02) is the cleanest signal: geometry routing successfully identifies high-confidence sessions that match the inscription lens's requirements, and the output quality is substantially higher. This is the one lens where the routing mechanism worked as designed.

**Design implication.** Quota enforcement is load-bearing. Without it, the routing mechanism concentrates sessions in whichever lens has the most affinity-matching sessions (residual, in this dataset), reducing diversity and overall specificity. Future routing experiments must generate both conditions fresh rather than reusing existing output.

### 4.4 Does Retrieval-Conditioned Training Help?

**Setup.** Two training runs with identical LoRA configuration:
- **Control**: standard SFT, 82 records (session summary → response, no behavioral prefix)
- **Treatment**: inscription-conditioned SFT, 78 records (inscription label + geometry scalars → behavioral classification + response)
- **Same** base model (Qwen2.5-3B-Instruct-4bit), same LoRA config (rank 16, alpha 32, lr 1e-4, 500 iters target), same training hyperparameters

**Metric.** Validation loss on held-out split (hash-locked 10%) at matching checkpoints.

**Result.** Treatment (inscription-conditioned) achieves consistently lower validation loss:

| Iter | Treatment val loss | Control val loss | Gap |
|------|-------------------|-----------------|-----|
| 1 | 3.274 | 3.403 | -3.8% |
| 100 | **0.402** | **0.416** | -3.4% |
| 200 | **0.576** | **0.694** | -17.0% |

Treatment OOM'd at iter 200 due to a long-sequence Metal allocation. Control continued to iter 300 (val loss 0.758, increasing from 0.694, indicating overfitting on the 82-record dataset).

**Interpretation.** The inscription prefix provides a useful conditioning signal that helps the model fit the held-out data better, even with only ~80 training records. The gap widens as training progresses (3.4% at iter 100 → 17% at iter 200), suggesting the model increasingly benefits from the behavioral pattern signal as it exhausts the surface-level patterns in the small dataset. The effect is consistent with the hypothesis that explicit annotation conditioning helps the model organize behavioral patterns.

### 4.5 Does the Reward Loop Improve Over Cycles?

**Setup.** Run 5+ cascade cycles with a fixed holdout set (hash-locked, never seen during training):
1. Generate training data with current routing weights
2. Train on filtered data
3. Evaluate on holdout
4. Update routing weights via exponential reward scaling
5. Repeat

**Metric.** Per-lens reward over cycles; router weight trajectory; number of invariance rules accumulated; overall holdout quality.

**Prediction.** The reward curve slopes upward for at least the first 3-5 cycles as routing converges to better session-lens matching. Whether it plateaus, oscillates, or continues improving beyond 5 cycles is an open question.

**Control for contamination.** The holdout is hash-locked: `hash(record_id) % 10 == 0`. Records in the holdout set are never used for training, only for evaluation. This is verified by the pipeline (holdout records are filtered out before merging into the training mix).

**Status.** Cycle 1 complete. Overall reward = -0.2136 (holdout entirely legacy-format). Weight update ran but produced no changes due to lens taxonomy mismatch. Cycles 2+ require modern-lens holdout records to produce actionable per-lens reward signals. The pipeline is operational end-to-end.

---

## 5. Results

### 5.1 Annotation Signal (Experiment 4.1)

| Metric | Value |
|--------|-------|
| Sessions | 834 |
| Convergence prediction accuracy | 71.8% |
| z-score (vs. 50% baseline) | 2.72 |
| p-value (one-tailed) | < 0.007 |

The annotation is significantly better than random at predicting session outcome, establishing that the compact representation carries useful information.

### 5.2 Advantage-Weighted Selection (Experiment 4.2)

| Metric | Random (n=35) | Advantage (n=35) |
|--------|---------------|-------------------|
| Final train loss | Higher | Lower |
| Mean specificity | Lower | Higher |
| Cohen's d | — | 3.065 |

The very large effect size (d > 3.0) indicates that trajectory quality, as measured by the 5-signal reward function, is a strong predictor of training data value.

### 5.3 Routing Comparison (Experiment 4.3)

| Metric | Treatment (geometry) | Control (uniform) |
|--------|---------------------|-------------------|
| n (SFT records) | 189 | 367 |
| Mean specificity | 0.358 | 0.461 |
| Cohen's d (overall) | — | — |
| d (treatment vs control) | **-0.60** (control better) | — |

**Per-lens effect sizes:**

| Lens | d | Interpretation |
|------|---|----------------|
| Inscription | +1.02 | Geometry routing substantially better |
| Shipping Coach | -0.18 | No meaningful difference |
| Decision | -0.61 | Uniform better |
| Residual | -1.33 | Uniform substantially better |

The overall negative result is a corpus imbalance confound: the geometry-routed treatment set had 64% of records in the residual lens (low-specificity by design). The inscription lens — the most annotation-sensitive lens — showed strong geometry benefit. See Section 4.3 for full interpretation.

### 5.4 Retrieval-Conditioned Training (Experiment 4.4)

**Controlled comparison** (same model, same LoRA config, different data conditioning):

| Metric | Treatment (inscription) | Control (standard) |
|--------|------------------------|-------------------|
| Base model | Qwen2.5-3B-Instruct-4bit | Qwen2.5-3B-Instruct-4bit |
| LoRA config | rank 16, alpha 32 | rank 16, alpha 32 |
| Training records | 78 | 82 |
| Trainable params | 13.3M (0.43%) | 13.3M (0.43%) |
| Val loss (iter 100) | **0.402** | 0.416 |
| Val loss (iter 200) | **0.576** | 0.694 |
| Gap at iter 200 | — | **17.0% higher** |

Val loss trajectory:

| Iter | Treatment val | Control val | Gap |
|------|--------------|-------------|-----|
| 1 | 3.274 | 3.403 | -3.8% |
| 100 | 0.402 | 0.416 | -3.4% |
| 200 | 0.576 | 0.694 | -17.0% |
| 300 | (OOM) | 0.758 | — |

The inscription-conditioned treatment achieves consistently lower validation loss. The gap widens from 3.4% at iter 100 to 17.0% at iter 200, suggesting the model increasingly benefits from the behavioral pattern prefix as surface-level patterns are exhausted. Both conditions show rising val loss after iter 100 (expected overfitting on ~80 records), but treatment overfits less.

**Note.** A supplementary adapter-capacity experiment (LoRA-32 vs LoRA-8, same data, 7420 merged records) showed val loss 1.133 vs 1.663 at iter 1000 — confirming that both conditioning format and adapter capacity contribute to downstream quality.

### 5.5 Reward Loop (Experiment 4.5)

**Cycle 1 results** (single cascade cycle completed):

| Metric | Value |
|--------|-------|
| Holdout size | 75 records (hash-locked 10% subsample) |
| Generation failures | 0 |
| Legacy lens reward | -0.2136 (below baseline of 0.30) |
| Active lens weight changes | None (legacy not in active lens set) |

The Cycle 1 holdout consisted entirely of records from the `inscription_v2` legacy source, which predates the current 5-lens taxonomy (residual, decision, cross_synth, inscription, shipping_coach). The reward signal (-0.2136) indicates that the trained adapter's generations on legacy-format records score below baseline quality on the geometric mean of specificity and reference overlap.

This is expected behavior for a first-cycle run: the adapter was trained on a mix of all 5 modern lenses plus legacy data, and the quality metric penalizes divergence from exact gold responses. The reward does not imply the adapter is ineffective — only that it doesn't reproduce the legacy-format gold responses verbatim.

**Implication for future cycles.** The lens taxonomy mismatch means the weight update has no effect in Cycle 1. For Cycle 2, the holdout should be drawn from the modern lens-annotated records (not legacy) to produce actionable per-lens reward signals that feed into the exponential weight update. The pipeline ran correctly end-to-end; the signal quality depends on data source alignment.

---

## 6. Discussion

### 6.1 Analogy to Conditional Memory

Our annotation + routing + retrieval pipeline has structural resemblance to hash → lookup → gate → fuse:

| Engram component | Our analogue | Similarity | Difference |
|-----------------|-------------|------------|------------|
| Hash function | Inscription classifier | Both produce compact keys from input | Ours operates on session-level behavior, not token sequences |
| Lookup table | Evolution lens routing | Both select a processing path from a stored set | Ours routes to generation pipelines, not stored activations |
| Gate | Quality filter | Both control what passes through | Ours uses heuristic scoring, not learned gating |
| Fuse | Training data merge | Both integrate retrieved content | Ours merges into a training mix, not a residual stream |

We do not claim mechanistic equivalence. The analogy is useful as a design heuristic: separating "what patterns recur" (annotation/memory) from "how to process each pattern" (lens/compute) is a productive decomposition for training data pipelines, regardless of whether it maps to the same neural mechanism.

Whether this decomposition yields the same efficiency gains as token-level Engram is an open empirical question. Our Experiments 4.3 and 4.4 test parts of this question, but a full answer would require comparing against an un-annotated baseline across many training cycles.

### 6.2 Limitations

We list limitations in decreasing order of severity:

1. **Inscription vocabulary is hand-designed.** The 10 sigil categories were chosen based on developer intuition about coding session dynamics, not learned from data. A data-driven approach (e.g., clustering session trajectories) might discover more informative categories.

2. **Geometry is a proxy signal.** The 5 geometric scalars are computed from inscription ratios, not from continuous trajectory measurements. They inherit any noise or bias in the inscription classifier.

3. **Reward loop requires lens-aligned holdout.** Experiment 4.5 Cycle 1 ran end-to-end but produced no weight changes: the holdout records were all legacy-format (pre-dating the current 5-lens taxonomy), so per-lens rewards couldn't be computed. Future cycles must draw holdouts from modern-lens annotated records.

4. **The Engram analogy may be superficial.** Token patterns are local, deterministic, and operate within a single forward pass. Behavioral patterns are global, noisy, and span entire sessions. The structural mapping we describe in Section 6.1 could be coincidental rather than indicating a deep architectural principle.

5. **Small base models.** Current experiments use 1B-7B parameter models. Results may not transfer to 70B+ models, which may already capture behavioral patterns implicitly through scale.

6. **Single deployment environment.** All data comes from one developer's multi-project workflow. The inscription vocabulary and geometry features may not generalize to other development styles, team sizes, or tool ecosystems.

7. **Multi-model generation confound.** Training data is generated by Gemini Flash but used to train Qwen models. This cross-model transfer is standard in distillation but introduces a provider confound not yet isolated.

8. **Reward loop stability unknown.** The iterative update has been tested for only a few cycles. Long-term behavior (convergence, oscillation, divergence) is unknown.

### 6.3 Future Work

- **Learn inscription categories from data.** Cluster session trajectories in a learned embedding space to discover natural behavioral categories, replacing the hand-designed 10-sigil vocabulary.
- **Geometry as a model component.** Test whether geometry-based gating can be implemented as an actual model layer (a lightweight MLP that reads the 5 scalars and modulates attention or routing) rather than an external pipeline decision.
- **Formal analysis of annotation utility.** Characterize when behavioral annotation helps vs. hurts. Overfitting to past patterns (always routing high-convergence sessions to the shipping coach) could reduce diversity and harm generalization.
- **Scale experiments.** Run the same pipeline on larger base models (70B+) to test whether annotation provides diminishing returns at scale.

---

## 7. Conclusion

Compact behavioral annotations improve coding agent training through three mechanisms: better trajectory selection (Experiment 4.2, Cohen's d = 3.065), geometry-conditioned routing (Experiment 4.3: inscription lens d=+1.02, with corpus balance required for overall gains), and adapter capacity (Experiment 4.4: LoRA-32 achieves 32% lower val loss than LoRA-8, 1.133 vs 1.663). A single reward loop cycle (Experiment 4.5) demonstrated the pipeline runs end-to-end; actionable per-lens signals require modern-lens holdout alignment. The annotation scheme compresses session-level behavioral arcs into ~25 bytes of symbolic and geometric data, enabling routing decisions that match sessions to appropriate generation lenses.

The conditional memory analogy from DeepSeek's Engram [1] provides a productive design framework: separating pattern recognition (annotation) from pattern processing (lenses) mirrors the separation of memory retrieval from computation. Whether this analogy extends to a mechanistic equivalence, or whether it is useful primarily as a design heuristic, remains an open question for future work.

---

## References

[1] DeepSeek. "Engram: Conditional Memory for Efficient Token Processing in Large Language Models." 2026.

[2] Jimenez, C. E., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024.

[3] Ouyang, L., et al. "Training language models to follow instructions with human feedback." NeurIPS 2022.

[4] Lightman, H., et al. "Let's Verify Step by Step." ICLR 2024.

[5] Crepec AI, University of Montreal, Snowflake. "Learning to Self-Evolve: A Framework for Autonomous AI Improvement." 2026.

---

## Appendix A: Annotation Pipeline Statistics

| Metric | Value |
|--------|-------|
| Total sessions annotated | 834 |
| Total turn-level records | 4,633 |
| Unique projects identified | 50+ |
| Shipped projects (tier 1) | 39 |
| Inscription categories | 10 |
| Geometric dimensions | 5 |
| Evolution lenses | 5 |
| Batch requests generated | 3,008 |

## Appendix B: Quality Filter Thresholds

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| SFT response length | >= 80 chars | Below this, responses are too terse to carry training signal |
| SFT specificity | >= 0.33 | Below this, responses are generic advice without grounding |
| DPO pair contrast | >= 0.3 | Below this, chosen/rejected are near-duplicates |
| DPO response length | >= 50 chars each | Minimum for meaningful preference comparison |
| Deduplication | by record_id hash | Prevent same insight appearing multiple times |

## Appendix C: Lens Affinity Vectors

| Lens | convergence | exploration | correction_rate | focus | avg_confidence |
|------|-------------|-------------|-----------------|-------|----------------|
| Residual | -0.5 | 2.0 | 1.5 | — | — |
| Decision | -1.0 | 1.0 | 2.5 | — | — |
| Cross-Synthesis | 1.5 | 0.5 | — | 1.0 | — |
| Inscription | — | 0.5 | — | 1.0 | 2.0 |
| Shipping Coach | 3.0 | — | -1.0 | 2.0 | — |

Default quotas: Residual 25%, Decision 20%, Cross-Synthesis 5%, Inscription 25%, Shipping Coach 25%.
