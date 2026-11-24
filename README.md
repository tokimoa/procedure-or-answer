# procedure-or-answer

Experiments that test which part of a retrieved memory an on-device language
model uses: the demonstrated procedure, or the stored answer. A model solves a
task with few-shot exemplars drawn from a local store, and the store is edited
one part at a time (answer field, procedure, fact record) to measure what
changes.

Everything runs locally on Apple silicon with 4-bit MLX models.

Paper: Takuya Omi, *Procedure or Answer? What Part of Memory Helps On-Device
Self-Improvement Depends on the Task*, Findings of EACL 2027.

## Installation

Python 3.12 on Apple silicon.

```bash
git clone https://github.com/tokimoa/procedure-or-answer.git
cd procedure-or-answer
pip install -r requirements.txt
```

Models are downloaded from the Hugging Face `mlx-community` organization on
first use. BBH, PopQA, and GSM8K are loaded with `datasets`.

## Quickstart

The repository ships the result files of every run, so the statistics and
figures can be recomputed without loading a model:

```bash
python experiments/analyze_paper_stats.py     # paired differences, CIs, equivalence tests
python experiments/make_paper_figures.py      # writes six figures to figures/
```

Run one experiment on one model (about 2 GB of memory, a few minutes):

```bash
python experiments/exp15_knowledge.py --models llama3.2-3b --n 24 --k 3 --seeds 1 \
    --out results/my_run.json
```

Run commands from the repository root.

## Repository layout

```
experiments/          one script per experiment, plus analysis and figure scripts
  models.py           registry of the ten models (four families, 1B to 30B)
  memory.py           few-shot prompt construction and solving with exemplars
  pools.py            synthetic tasks, exemplar pools, answer-field corruption
rsiguard/backends.py  MLX inference wrapper: seeded decoding, token embeddings
results/              result files (JSON), one per run
run_experiments.sh    the command behind each result file
```

## Experiments

| Script | What it measures | Result file |
|---|---|---|
| `exp13_ingredient.py` | Procedural tasks: worked solution, bare answer, corrupted answer | `exp13_ci_v2.json`, `exp13_roster_v2.json` |
| `exp18_procedure.py` | Corrupting the method versus the answer | `exp18_ci_v2.json` |
| `exp26_ownstyle_method.py` | Method corruption with the model's own-style exemplars | `exp26_ownstyle.json`, `exp26_decoupled.json` |
| `exp23_copyrate.py`, `exp25_copyrate_k3.py` | Copy-rate of a planted wrong answer or fact | `exp23_copyrate_v2.json`, `exp25_copyrate_k3.json` |
| `exp24_unfiltered.py`, `exp24b_wrongonly.py` | Unfiltered self-generated pools; wrong-only pools | `exp24_unfiltered.json`, `exp24b_wrongonly.json` |
| `exp28_procedure_only.py`, `probe_proconly_format.py` | Removing the stored answer; output-format probe | `exp28_proc_only.json`, `exp28_bbh_proc_only.json`, `exp28_format_probe*.json` |
| `exp15_knowledge.py`, `exp16_knowledge_rho.py` | Knowledge lookup; fraction of corrupted facts | `exp15_ci_v2.json`, `exp16_ci_v2.json` |
| `exp21_mixed.py` | Task needing both a stored fact and a procedure | `exp21_mixed.json` |
| `exp17_ksens.py` | Sensitivity to the number of retrieved exemplars | `exp17_ksens.json` |
| `exp34_two_fact.py` | Queries that combine two stored facts | `exp34_two_fact.json` |
| `exp22_gsm8k.py` | BBH multistep arithmetic, BBH object counting, GSM8K | `exp22_bbh_ci.json`, `exp22_objcount_ci.json`, `exp22_gsm8k.json` |
| `exp27_bbh_nonnumeric.py` | BBH state tracking and Boolean expressions | `exp27_tracking3.json`, `exp27_boolean.json` |
| `exp19_popqa.py`, `probe_nomem_prompt.py` | PopQA long-tail questions; zero-shot prompt control | `exp19_ci.json`, `probe_nomem_prompt.json` |
| `exp30_bbh_fixed_loo.py` | All 250 BBH items with leave-one-out retrieval | `exp30_bbh_fixed.json` |
| `exp29_popqa_nonoracle_fixed.py` | 175 PopQA questions with same-entity distractors | `exp29_popqa_fixed.json` |
| `exp31_filtered_pool_audit.py`, `audit_filtered_pool.py`, `finalize_filtered_pool_audit.py` | Rule-based audit of answer-correct exemplars | `exp31_filtered_audit_*.json` |
| `exp32_screened_pool_sensitivity.py` | Accuracy with flagged exemplars removed | `exp32_screened_only.json` |
| `exp33_working_value_mask.py` | Masking the derived value inside the working | `exp33_working_mask.json` |

Analysis scripts: `analyze_paper_stats.py` (main statistics),
`analyze_rebuttal_addons.py` (exp29, exp30), `analyze_screened_sensitivity.py`
(exp32), `analyze_two_fact.py` (exp34). The last three take result-file paths
as arguments.

## Reproducing the runs

`run_experiments.sh` lists the command that produced each result file. Run a
single block, or the whole file:

```bash
sh run_experiments.sh
```

- The full set takes many hours on one laptop. The paper's runs used 36 GB of
  unified memory; the four main models need under 6 GB each.
- Decoding is sampled at temperature 0.7 and seeded per seed and item. Exact
  numbers can still differ across machines and across re-downloads of the
  model weights.
- `exp29` to `exp34` load one model at a time and checkpoint as they go, so an
  interrupted run resumes with the same arguments.

## Result files

- Each file holds per-model, per-seed, per-condition accuracies. Some also
  hold the synthetic problems and the models' generations on them.
- No benchmark items are redistributed. Files for BBH, PopQA, and GSM8K hold
  accuracies only; `exp29_popqa_fixed.json` and `exp30_bbh_fixed.json` hold
  per-item correctness and retrieval hits keyed by the benchmarks' own item
  ids.
- The `meta` fields of `exp32_screened_only.json` and
  `exp33_working_mask.json` record run-time file names that carried a
  `rebuttal_` prefix.

## License

Code: MIT (`LICENSE`). Result files: CC BY 4.0 (`results/LICENSE`).
BBH, PopQA, GSM8K, and the models keep their own licenses.

## Citation

```bibtex
@inproceedings{omi2027procedure,
  title     = {Procedure or Answer? What Part of Memory Helps On-Device Self-Improvement Depends on the Task},
  author    = {Omi, Takuya},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
