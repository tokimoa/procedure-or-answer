# Procedure or Answer? What Part of Memory Helps On-Device Self-Improvement Depends on the Task

Code and per-run result files for the paper

> Takuya Omi. *Procedure or Answer? What Part of Memory Helps On-Device
> Self-Improvement Depends on the Task.* Findings of EACL 2027.

A language model can improve without weight updates by storing its own solved
problems and retrieving them as few-shot context. The paper asks which part of
a stored exemplar carries the gain. On procedural (recomputable) tasks it is
the demonstrated procedure: corrupting the stored answer costs almost nothing,
because the model recomputes it. On knowledge-lookup tasks it is the stored
fact: corrupting it removes all of memory's value.

This repository contains every experiment script, the shared inference
backend, and the result files behind every table, figure, and statistic in the
paper. All experiments run on one Apple-silicon laptop with 4-bit MLX models.

## Contents

```
experiments/   one script per experiment, plus the analysis and figure scripts
rsiguard/      on-device inference backend (mlx-lm): seeded decoding, embeddings
results/       per-run result files (JSON), one per experiment run
```

Shared modules in `experiments/`:

- `models.py`: registry of the ten models used (four families, 1B to 30B)
- `memory.py`: few-shot prompt construction and solving with retrieved exemplars
- `pools.py`: synthetic procedural tasks, exemplar-pool construction, answer-field corruption

## Recomputing the paper's numbers

No model is needed for this step; it reads `results/` only.

```bash
pip install -r requirements.txt

python experiments/analyze_paper_stats.py      # statistics of Sections 4 and 5, Appendices B to I
python experiments/make_paper_figures.py       # all six figures, written to figures/
python experiments/analyze_rebuttal_addons.py results/exp29_popqa_fixed.json results/exp30_bbh_fixed.json
python experiments/analyze_screened_sensitivity.py results/exp32_screened_only.json
python experiments/analyze_two_fact.py results/exp34_two_fact.json
```

Run every command from the repository root.

## Result-to-file map

This is Table 18 (Appendix M) of the paper.

| Result | Script (`experiments/`) | Result file (`results/`) |
|---|---|---|
| Table 1; Section 4.1 | `exp13_ingredient.py` | `exp13_ci_v2.json` |
| Table 6 (ten-model roster) | `exp13_ingredient.py` | `exp13_roster_v2.json` |
| Table 2; Figure 5 | `exp18_procedure.py` | `exp18_ci_v2.json` |
| Table 7 (own-style replication) | `exp26_ownstyle_method.py` | `exp26_ownstyle.json`, `exp26_decoupled.json` |
| Figure 1 (copy-rate, k=3) | `exp25_copyrate_k3.py` | `exp25_copyrate_k3.json` |
| Table 10 (copy-rate, k=1) | `exp23_copyrate.py` | `exp23_copyrate_v2.json` |
| Table 8 (unfiltered pools) | `exp24_unfiltered.py` | `exp24_unfiltered.json` |
| Section 4.4 (wrong-only control) | `exp24b_wrongonly.py` | `exp24b_wrongonly.json` |
| Table 3; Figure 4 | `exp15_knowledge.py` | `exp15_ci_v2.json` |
| Figure 3 (pollution sweep) | `exp16_knowledge_rho.py` | `exp16_ci_v2.json` |
| Figure 6 (compositional test) | `exp21_mixed.py` | `exp21_mixed.json` |
| Table 4, BBH rows; Figure 2 | `exp22_gsm8k.py` | `exp22_bbh_ci.json` |
| Table 4, PopQA rows; Figure 2 | `exp19_popqa.py` | `exp19_ci.json` |
| Table 13 (object counting) | `exp22_gsm8k.py` | `exp22_objcount_ci.json` |
| Tables 11, 12 (state tracking, Boolean) | `exp27_bbh_nonnumeric.py` | `exp27_tracking3.json`, `exp27_boolean.json` |
| Table 14 (GSM8K) | `exp22_gsm8k.py` | `exp22_gsm8k.json` |
| Table 9 (answer removal) | `exp28_procedure_only.py` | `exp28_proc_only.json`, `exp28_bbh_proc_only.json` |
| Appendix E (format probe) | `probe_proconly_format.py` | `exp28_format_probe.json`, `exp28_format_probe_texts.json` |
| Appendix A (zero-shot prompt control) | `probe_nomem_prompt.py` | `probe_nomem_prompt.json` |
| k-sensitivity (Section 3) | `exp17_ksens.py` | `exp17_ksens.json` |
| Table 5, BBH; Appendix J | `exp30_bbh_fixed_loo.py` | `exp30_bbh_fixed.json` |
| Tables 5 and 15, PopQA | `exp29_popqa_nonoracle_fixed.py` | `exp29_popqa_fixed.json` |
| Table 16, audit | `exp31_filtered_pool_audit.py`, `audit_filtered_pool.py`, `finalize_filtered_pool_audit.py` | `exp31_filtered_audit_diagnostics_v3.json`, `exp31_filtered_audit_final.json` |
| Table 16, screened | `exp32_screened_pool_sensitivity.py` | `exp32_screened_only.json` |
| Table 16, masked | `exp33_working_value_mask.py` | `exp33_working_mask.json` |
| Table 17 (multi-fact queries) | `exp34_two_fact.py` | `exp34_two_fact.json` |

## Re-running the experiments

Requirements: Apple silicon (the paper used 36 GB of unified memory; the four
main models need under 6 GB each, and the ten-model roster goes up to 30B
parameters), Python 3.12, and the pinned versions in `requirements.txt`. Models are
downloaded from the Hugging Face `mlx-community` organization on first use;
BBH, PopQA, and GSM8K are loaded with `datasets`.

```bash
python experiments/exp15_knowledge.py \
    --models llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b \
    --n 24 --k 3 --seeds 5 --out results/my_exp15.json
```

The settings used in the paper are in its Appendix A: 5 seeds (3 for GSM8K),
24 test items per condition per seed (25 for BBH), k=3 retrieved exemplars,
temperature 0.7 with per-seed reseeding. The usage lines in the script
docstrings are examples and often use fewer seeds. The later scripts (`exp29`
to `exp34`) load one model at a time and checkpoint as they go, so an
interrupted run resumes with the same arguments.

Sampled decoding is seeded, but exact numbers can differ across machines and
across re-downloads of the model weights. The paired contrasts are stable.

The audit pipeline runs in three steps: `exp31_filtered_pool_audit.py`
regenerates and archives the answer-correct exemplars,
`audit_filtered_pool.py` adds the rule-based diagnostics, and
`finalize_filtered_pool_audit.py` applies the two rules and writes the final
file. `prepare_filtered_audit_sheet.py` builds the sheet used to check the
flagged equations by hand.

## About the result files

- Each file holds per-model, per-seed, per-condition accuracies. Some also
  hold our own synthetic problems and the models' generations on them (the
  unfiltered pools, the audit population, the copy-rate transcripts).
- No benchmark items are redistributed. Files for BBH, PopQA, and GSM8K hold
  accuracies only. The two fixed-set files hold per-item correctness and
  retrieval hits keyed by the benchmarks' own item ids, without question
  text, gold answers, or model outputs.
- No personal data is included.
- The `meta` fields of `exp32_screened_only.json` and `exp33_working_mask.json`
  record the file names used at run time, which carried a `rebuttal_` prefix.
  The files are otherwise unchanged.

## License

Code: MIT (`LICENSE`). Result files: CC BY 4.0 (`results/LICENSE`).
BBH, PopQA, GSM8K, and the models keep their own licenses.

## Use of AI assistants

AI tools assisted with coding: the implementation follows the author's
designs, with assistants used for review and suggestions rather than wholesale
generation. All experimental designs, analyses, and numbers were verified by
the author against the recorded run logs.

## Citation

```bibtex
@inproceedings{omi2027procedure,
  title     = {Procedure or Answer? What Part of Memory Helps On-Device Self-Improvement Depends on the Task},
  author    = {Omi, Takuya},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
