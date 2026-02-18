"""
Progress report: shows pipeline status and viability assessment
for a methods paper on multi-run LLM annotation reliability.

Usage:
    python progress_report.py                    # latest experiment
    python progress_report.py --experiment-id 3  # specific experiment
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import argparse
from config import DB_CONFIG

# Thresholds for a publishable methods paper
VIABILITY = {
    'min_stability_rate': 0.60,       # 60%+ chunks stable
    'min_krippendorff_alpha': 0.40,   # fair+ agreement (0.67 is good)
    'min_clarity_rate': 0.50,         # 50%+ responses parseable
    'min_valid_response_rate': 0.80,  # 80%+ responses non-empty
    'target_stability_rate': 0.80,    # ideal for publication
    'target_alpha': 0.67,             # standard "good" threshold
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def report_experiment_info(cur, exp_id):
    print_section("EXPERIMENT INFO")
    cur.execute("""
        SELECT experiment_id, experiment_name, status, num_chunks_tested,
               models_tested, temperatures_tested, num_runs,
               started_at, completed_at
        FROM experiment_runs WHERE experiment_id = %s
    """, (exp_id,))
    row = cur.fetchone()
    if not row:
        print(f"  No experiment found with ID {exp_id}")
        return False
    print(f"  ID:           {row['experiment_id']}")
    print(f"  Name:         {row['experiment_name']}")
    print(f"  Status:       {row['status']}")
    print(f"  Chunks:       {row['num_chunks_tested']}")
    print(f"  Models:       {row['models_tested']}")
    print(f"  Temperatures: {row['temperatures_tested']}")
    print(f"  Runs/task:    {row['num_runs']}")
    print(f"  Started:      {row['started_at']}")
    print(f"  Completed:    {row['completed_at'] or 'still running'}")
    return True


def report_task_progress(cur, exp_id):
    print_section("1. PIPELINE PROGRESS")
    cur.execute("""
        SELECT status, COUNT(*) AS cnt
        FROM annotation_tasks
        WHERE experiment_id = %s
        GROUP BY status ORDER BY status
    """, (exp_id,))
    rows = cur.fetchall()
    total = sum(r['cnt'] for r in rows)
    done = sum(r['cnt'] for r in rows if r['status'] == 'completed')
    failed = sum(r['cnt'] for r in rows if r['status'] == 'failed')
    pct = (done / total * 100) if total else 0
    for r in rows:
        print(f"  {r['status']:12s} {r['cnt']:>6,}")
    print(f"  {'TOTAL':12s} {total:>6,}")
    print(f"\n  Completion: {done:,}/{total:,} ({pct:.1f}%)")
    if failed:
        print(f"  WARNING: {failed:,} failed tasks")
    return done, total


def report_response_quality(cur, exp_id):
    print_section("2. RESPONSE QUALITY (are models returning usable output?)")
    cur.execute("""
        SELECT
            model_name,
            COUNT(*) AS total,
            SUM(CASE WHEN raw_response = '' THEN 1 ELSE 0 END) AS empty,
            SUM(CASE WHEN label_kind = 'unclear' THEN 1 ELSE 0 END) AS unclear,
            SUM(CASE WHEN label_kind IN ('float', 'category') THEN 1 ELSE 0 END) AS valid,
            SUM(CASE WHEN label_kind = 'none' THEN 1 ELSE 0 END) AS none_label
        FROM llm_annotation_runs
        WHERE experiment_id = %s
        GROUP BY model_name ORDER BY model_name
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No annotation results yet.")
        return 0.0

    total_all = 0
    valid_all = 0
    empty_all = 0
    print(f"  {'Model':<30s} {'Total':>6} {'Valid':>6} {'Empty':>6} {'Unclear':>8} {'None':>6} {'Valid%':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")
    for r in rows:
        total_all += r['total']
        valid_all += r['valid']
        empty_all += r['empty']
        vr = (r['valid'] / r['total'] * 100) if r['total'] else 0
        print(f"  {r['model_name']:<30s} {r['total']:>6} {r['valid']:>6} {r['empty']:>6} {r['unclear']:>8} {r['none_label']:>6} {vr:>6.1f}%")

    overall_valid_rate = (valid_all / total_all) if total_all else 0
    overall_empty_rate = (empty_all / total_all) if total_all else 0
    print(f"\n  Overall valid response rate: {overall_valid_rate:.1%}")
    if overall_empty_rate > 0.1:
        print(f"  WARNING: {overall_empty_rate:.1%} of responses are EMPTY (model not generating output)")
    return overall_valid_rate


def report_response_quality_by_construct(cur, exp_id):
    print_section("3. RESPONSE QUALITY BY CONSTRUCT")
    cur.execute("""
        SELECT
            construct_name,
            COUNT(*) AS total,
            SUM(CASE WHEN label_kind IN ('float', 'category') THEN 1 ELSE 0 END) AS valid,
            SUM(CASE WHEN label_kind = 'none' THEN 1 ELSE 0 END) AS none_label,
            SUM(CASE WHEN label_kind = 'unclear' THEN 1 ELSE 0 END) AS unclear
        FROM llm_annotation_runs
        WHERE experiment_id = %s
        GROUP BY construct_name ORDER BY construct_name
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No data yet.")
        return
    print(f"  {'Construct':<25s} {'Total':>6} {'Valid':>6} {'None':>6} {'Unclear':>8} {'Valid%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
    for r in rows:
        vr = (r['valid'] / r['total'] * 100) if r['total'] else 0
        print(f"  {r['construct_name']:<25s} {r['total']:>6} {r['valid']:>6} {r['none_label']:>6} {r['unclear']:>8} {vr:>6.1f}%")


def report_stability(cur, exp_id):
    print_section("4. STABILITY (do repeated runs agree?)")
    cur.execute("""
        SELECT
            construct_name,
            model_name,
            COUNT(*) AS num_chunks,
            SUM(CASE WHEN is_stable THEN 1 ELSE 0 END) AS stable,
            ROUND(AVG(CASE WHEN is_stable THEN 1.0 ELSE 0.0 END)::numeric, 3) AS stability_rate,
            ROUND(AVG(agreement_ratio)::numeric, 3) AS avg_agreement,
            ROUND(AVG(stdev_value)::numeric, 3) AS avg_stdev
        FROM annotation_stability_metrics
        WHERE experiment_id = %s
        GROUP BY construct_name, model_name
        ORDER BY construct_name, model_name
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No stability data yet (pipeline hasn't reached step 5).")
        return None

    print(f"  {'Construct':<25s} {'Model':<25s} {'Chunks':>6} {'Stable':>6} {'Rate':>7} {'Agree':>7} {'StDev':>7}")
    print(f"  {'-'*25} {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        sr = r['stability_rate'] or 0
        ag = r['avg_agreement'] if r['avg_agreement'] is not None else ''
        sd = r['avg_stdev'] if r['avg_stdev'] is not None else ''
        print(f"  {r['construct_name']:<25s} {r['model_name']:<25s} {r['num_chunks']:>6} {r['stable']:>6} {sr:>7} {ag:>7} {sd:>7}")
    return rows


def report_group_reliability(cur, exp_id):
    print_section("5. GROUP-LEVEL RELIABILITY (Krippendorff alpha / ICC)")
    cur.execute("""
        SELECT
            construct_name,
            model_name,
            temperature,
            krippendorff_alpha,
            icc_value,
            stability_rate,
            coverage_rate,
            clarity_rate
        FROM group_reliability_metrics
        WHERE experiment_id = %s
        ORDER BY construct_name, model_name, temperature
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No group reliability data yet (pipeline hasn't reached step 5).")
        return None

    print(f"  {'Construct':<22s} {'Model':<22s} {'Temp':>5} {'Alpha':>7} {'ICC':>7} {'Stab%':>7} {'Cover':>7} {'Clear':>7}")
    print(f"  {'-'*22} {'-'*22} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        alpha = f"{r['krippendorff_alpha']:.3f}" if r['krippendorff_alpha'] is not None else '  n/a'
        icc = f"{r['icc_value']:.3f}" if r['icc_value'] is not None else '  n/a'
        stab = f"{r['stability_rate']:.3f}" if r['stability_rate'] is not None else '  n/a'
        cov = f"{r['coverage_rate']:.3f}" if r['coverage_rate'] is not None else '  n/a'
        clar = f"{r['clarity_rate']:.3f}" if r['clarity_rate'] is not None else '  n/a'
        print(f"  {r['construct_name']:<22s} {r['model_name']:<22s} {r['temperature']:>5.1f} {alpha:>7} {icc:>7} {stab:>7} {cov:>7} {clar:>7}")
    return rows


def report_temperature_effect(cur, exp_id):
    print_section("6. TEMPERATURE EFFECT (0.0 vs 0.5)")
    cur.execute("""
        SELECT
            temperature,
            ROUND(AVG(stability_rate)::numeric, 3) AS avg_stability,
            ROUND(AVG(krippendorff_alpha)::numeric, 3) AS avg_alpha,
            ROUND(AVG(clarity_rate)::numeric, 3) AS avg_clarity,
            COUNT(*) AS conditions
        FROM group_reliability_metrics
        WHERE experiment_id = %s
        GROUP BY temperature
        ORDER BY temperature
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No data yet.")
        return
    for r in rows:
        print(f"  temp={r['temperature']:.1f}: stability={r['avg_stability']} alpha={r['avg_alpha']} clarity={r['avg_clarity']} (n={r['conditions']})")


def report_sample_responses(cur, exp_id):
    print_section("7. SAMPLE RESPONSES (first 5 non-empty)")
    cur.execute("""
        SELECT model_name, construct_name, label_kind, label_value_text,
               label_value_float, raw_response
        FROM llm_annotation_runs
        WHERE experiment_id = %s AND LENGTH(raw_response) > 0
        ORDER BY run_id
        LIMIT 5
    """, (exp_id,))
    rows = cur.fetchall()
    if not rows:
        print("  No non-empty responses found. Models may not be generating output.")
        # Show a few empty ones to confirm
        cur.execute("""
            SELECT model_name, construct_name, label_kind, raw_response
            FROM llm_annotation_runs
            WHERE experiment_id = %s
            ORDER BY run_id
            LIMIT 3
        """, (exp_id,))
        empty_rows = cur.fetchall()
        if empty_rows:
            print("  Sample empty responses:")
            for r in empty_rows:
                print(f"    [{r['model_name']}] {r['construct_name']}: raw={repr(r['raw_response'][:100])}")
        return

    for r in rows:
        raw = r['raw_response'][:120].replace('\n', ' ')
        print(f"  [{r['model_name']}] {r['construct_name']}")
        print(f"    Parsed: kind={r['label_kind']} text={r['label_value_text']} float={r['label_value_float']}")
        print(f"    Raw: {raw}")
        print()


def report_viability(cur, exp_id, valid_rate, stability_rows, reliability_rows):
    print_section("8. METHODS PAPER VIABILITY ASSESSMENT")

    checks = []

    # Check 1: Valid response rate
    ok = valid_rate >= VIABILITY['min_valid_response_rate']
    status = 'PASS' if ok else 'FAIL'
    checks.append(ok)
    print(f"  [{status}] Valid response rate: {valid_rate:.1%} (need >= {VIABILITY['min_valid_response_rate']:.0%})")

    # Check 2: Stability rate
    if stability_rows:
        rates = [float(r['stability_rate'] or 0) for r in stability_rows]
        avg_stab = sum(rates) / len(rates) if rates else 0
        ok = avg_stab >= VIABILITY['min_stability_rate']
        status = 'PASS' if ok else 'FAIL'
        checks.append(ok)
        target_ok = avg_stab >= VIABILITY['target_stability_rate']
        extra = ' (publication-quality!)' if target_ok else ''
        print(f"  [{status}] Avg stability rate: {avg_stab:.3f} (need >= {VIABILITY['min_stability_rate']:.2f}){extra}")
    else:
        print(f"  [WAIT] Stability: not computed yet")

    # Check 3: Krippendorff alpha
    if reliability_rows:
        alphas = [r['krippendorff_alpha'] for r in reliability_rows if r['krippendorff_alpha'] is not None]
        if alphas:
            avg_alpha = sum(alphas) / len(alphas)
            ok = avg_alpha >= VIABILITY['min_krippendorff_alpha']
            status = 'PASS' if ok else 'FAIL'
            checks.append(ok)
            target_ok = avg_alpha >= VIABILITY['target_alpha']
            extra = ' (publication-quality!)' if target_ok else ''
            print(f"  [{status}] Avg Krippendorff alpha: {avg_alpha:.3f} (need >= {VIABILITY['min_krippendorff_alpha']:.2f}){extra}")
        else:
            print(f"  [WAIT] Alpha: all values null (categorical constructs may use other metrics)")
    else:
        print(f"  [WAIT] Reliability: not computed yet")

    # Check 4: Clarity rate
    if reliability_rows:
        clarities = [r['clarity_rate'] for r in reliability_rows if r['clarity_rate'] is not None]
        if clarities:
            avg_clarity = sum(clarities) / len(clarities)
            ok = avg_clarity >= VIABILITY['min_clarity_rate']
            status = 'PASS' if ok else 'FAIL'
            checks.append(ok)
            print(f"  [{status}] Avg clarity rate: {avg_clarity:.3f} (need >= {VIABILITY['min_clarity_rate']:.2f})")

    # Summary
    print()
    if not checks:
        print("  VERDICT: Too early to assess. Pipeline still running.")
    elif all(checks):
        print("  VERDICT: VIABLE - Results support a publishable methods paper.")
    elif sum(checks) / len(checks) >= 0.5:
        print("  VERDICT: PROMISING - Some metrics pass. May need tuning or more data.")
    else:
        print("  VERDICT: NOT YET VIABLE - Key metrics below threshold.")
        print("  Consider: different models, prompt tuning, more chunks, or different constructs.")


def main():
    parser = argparse.ArgumentParser(description="Methods paper progress report")
    parser.add_argument('--experiment-id', type=int, default=None,
                        help='Experiment ID (default: latest)')
    args = parser.parse_args()

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if args.experiment_id is None:
        cur.execute("SELECT MAX(experiment_id) FROM experiment_runs")
        args.experiment_id = cur.fetchone()['max']
        if args.experiment_id is None:
            print("No experiments found.")
            return

    print(f"\n  METHODS PAPER PROGRESS REPORT")
    print(f"  Experiment {args.experiment_id}")

    if not report_experiment_info(cur, args.experiment_id):
        return

    done, total = report_task_progress(cur, args.experiment_id)
    valid_rate = report_response_quality(cur, args.experiment_id)
    report_response_quality_by_construct(cur, args.experiment_id)
    stability_rows = report_stability(cur, args.experiment_id)
    reliability_rows = report_group_reliability(cur, args.experiment_id)
    report_temperature_effect(cur, args.experiment_id)
    report_sample_responses(cur, args.experiment_id)
    report_viability(cur, args.experiment_id, valid_rate, stability_rows, reliability_rows)

    conn.close()


if __name__ == '__main__':
    main()
