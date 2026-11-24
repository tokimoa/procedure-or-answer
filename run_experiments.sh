#!/bin/sh
# Commands that produce the files in results/. Each block can be run on its own.
# Commands marked [recorded] are copied from the original batch scripts.
# Commands marked [settings] are rebuilt from the settings stored in the result
# file or from the script defaults that match it.
set -e
PY=${PY:-python}
MAIN="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b"
ROSTER="llama3.2-1b,llama3.2-3b,llama3.1-8b,phi4-mini,phi4,ministral3-3b,ministral3-8b,ministral3-14b,granite4.1-8b,granite4.1-30b"

# --- controlled procedural tasks ---------------------------------------------
# [recorded] exp13_ci_v2.json
$PY experiments/exp13_ingredient.py --models $MAIN --tasks arith_hard,invoice \
    --pool 24 --test 24 --k 3 --seeds 5 --max-tokens 384 --out results/exp13_ci_v2.json
# [recorded] exp13_roster_v2.json (ten models, 2 seeds)
$PY experiments/exp13_ingredient.py --models $ROSTER --tasks arith_hard,invoice \
    --pool 24 --test 24 --k 3 --seeds 2 --max-tokens 384 --out results/exp13_roster_v2.json
# [recorded] exp18_ci_v2.json
$PY experiments/exp18_procedure.py --models llama3.2-3b,phi4-mini,granite4.1-8b,phi4,ministral3-14b \
    --test 24 --seeds 5 --out results/exp18_ci_v2.json
# [recorded] exp26_ownstyle.json
$PY experiments/exp26_ownstyle_method.py --models $MAIN \
    --pooln 24 --cand-cap 90 --test 24 --k 3 --seeds 5 --max-tokens 256 --out results/exp26_ownstyle.json
# [settings] exp26_decoupled.json
$PY experiments/exp26_ownstyle_method.py --models ministral3-3b,granite4.1-8b --decouple \
    --pooln 24 --cand-cap 90 --test 24 --k 3 --seeds 5 --max-tokens 256 --out results/exp26_decoupled.json
# [recorded] exp24_unfiltered.json, then [settings] exp24b_wrongonly.json
$PY experiments/exp24_unfiltered.py --models $MAIN --tasks arith_hard,invoice \
    --pool 24 --test 24 --k 3 --seeds 5 --max-tokens 384 --out results/exp24_unfiltered.json
$PY experiments/exp24b_wrongonly.py --in results/exp24_unfiltered.json --out results/exp24b_wrongonly.json
# [recorded] exp28_proc_only.json, exp28_bbh_proc_only.json
$PY experiments/exp28_procedure_only.py --regime proc --models $MAIN \
    --tasks arith_hard,invoice --pool 24 --test 24 --k 3 --seeds 5 --max-tokens 384 --out results/exp28_proc_only.json
$PY experiments/exp28_procedure_only.py --regime bbh --models $MAIN \
    --tasks multistep_arithmetic_two --pool 60 --test 25 --k 3 --seeds 5 --max-tokens 384 --out results/exp28_bbh_proc_only.json
# [settings] exp28_format_probe.json
$PY experiments/probe_proconly_format.py --model phi4-mini --tasks invoice,arith_hard --seeds 2 \
    --out results/exp28_format_probe.json

# --- knowledge lookup, copy-rate, compositional task --------------------------
# [recorded] exp15_ci_v2.json, exp16_ci_v2.json
$PY experiments/exp15_knowledge.py --models $MAIN --seeds 5 --out results/exp15_ci_v2.json
$PY experiments/exp16_knowledge_rho.py --models $MAIN --n 24 --k 3 --seeds 5 --out results/exp16_ci_v2.json
# [recorded] exp23_copyrate_v2.json, exp25_copyrate_k3.json
$PY experiments/exp23_copyrate.py --models $MAIN --n 30 --seeds 5 --out results/exp23_copyrate_v2.json
$PY experiments/exp25_copyrate_k3.py --models $MAIN --n 24 --k 3 --seeds 5 --out results/exp25_copyrate_k3.json
# [recorded] exp21_mixed.json
$PY experiments/exp21_mixed.py --models $MAIN --n 24 --k 3 --seeds 5 --max-tokens 200 --out results/exp21_mixed.json
# [settings] exp17_ksens.json
$PY experiments/exp17_ksens.py --models phi4-mini,granite4.1-8b --ks 1,3,5,8 --seeds 2 --out results/exp17_ksens.json
# [settings] exp34_two_fact.json
$PY experiments/exp34_two_fact.py --out results/exp34_two_fact.json

# --- public benchmarks ---------------------------------------------------------
# [recorded] exp22_bbh_ci.json, exp22_objcount_ci.json, exp22_gsm8k.json
$PY experiments/exp22_gsm8k.py --dataset bbh --models $MAIN \
    --n-pool 60 --n-test 25 --k 3 --seeds 5 --max-tokens 384 --out results/exp22_bbh_ci.json
$PY experiments/exp22_gsm8k.py --dataset bbh --bbh-task object_counting --models $MAIN \
    --n-pool 60 --n-test 25 --k 3 --seeds 5 --max-tokens 384 --out results/exp22_objcount_ci.json
$PY experiments/exp22_gsm8k.py --models $MAIN \
    --n-pool 40 --n-test 24 --k 3 --seeds 3 --max-tokens 384 --out results/exp22_gsm8k.json
# [recorded] exp27_tracking3.json, exp27_boolean.json
$PY experiments/exp27_bbh_nonnumeric.py --models $MAIN --bbh-task tracking_shuffled_objects_three_objects \
    --n-pool 60 --n-test 25 --k 3 --seeds 5 --max-tokens 384 --out results/exp27_tracking3.json
$PY experiments/exp27_bbh_nonnumeric.py --models $MAIN --bbh-task boolean_expressions \
    --n-pool 60 --n-test 25 --k 3 --seeds 5 --max-tokens 256 --out results/exp27_boolean.json
# [recorded] exp19_ci.json, probe_nomem_prompt.json
$PY experiments/exp19_popqa.py --models $MAIN --n 24 --k 3 --seeds 5 --max-tokens 60 --out results/exp19_ci.json
$PY experiments/probe_nomem_prompt.py --models llama3.2-3b,ministral3-3b --n-test 25 --seeds 5 \
    --out results/probe_nomem_prompt.json
# [settings] exp29_popqa_fixed.json, exp30_bbh_fixed.json (script defaults)
# The released exp29/exp30 files have question text, gold answers and model outputs removed.
$PY experiments/exp29_popqa_nonoracle_fixed.py --out results/exp29_popqa_fixed.json
$PY experiments/exp30_bbh_fixed_loo.py --out results/exp30_bbh_fixed.json

# --- exemplar audit ---------------------------------------------------------------
# [settings] exp31 -> diagnostics -> final, then screened pools and working-value masking
$PY experiments/exp31_filtered_pool_audit.py --out results/exp31_filtered_audit.json
$PY experiments/audit_filtered_pool.py results/exp31_filtered_audit.json \
    --out results/exp31_filtered_audit_diagnostics_v3.json
$PY experiments/finalize_filtered_pool_audit.py results/exp31_filtered_audit_diagnostics_v3.json \
    --out results/exp31_filtered_audit_final.json
$PY experiments/exp32_screened_pool_sensitivity.py --conditions screened --out results/exp32_screened_only.json
$PY experiments/exp33_working_value_mask.py --out results/exp33_working_mask.json
