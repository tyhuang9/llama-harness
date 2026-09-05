"""Run and reduce the bounded local-task-agent strategy benchmark.

The driver is intentionally a one-process, standard-library-only harness.  It
does not start a model server, change provider configuration, or turn a failed
or missing child report into a passing latency sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


BENCHMARK_SUITE_ID = "local-task-agent-strategy-benchmark"
CASES = ("dependent-lookup-update", "independent-reads-8")
STRATEGIES = ("direct", "adaptive")
MEASURED_PAIRS_PER_CASE = 30


@dataclass(frozen=True)
class BenchmarkConfig:
    binary: Path
    suite: Path
    environment_json: Path
    source_root: Path
    output_dir: Path
    model: str
    ollama_url: str
    cohort_wall_seconds: float = 40 * 60


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = BenchmarkConfig(
        binary=arguments.binary.resolve(),
        suite=arguments.suite.resolve(),
        environment_json=arguments.environment_json.resolve(),
        source_root=arguments.source_root.resolve(),
        output_dir=arguments.output_dir.resolve(),
        model=arguments.model,
        ollama_url=arguments.ollama_url,
        cohort_wall_seconds=arguments.cohort_wall_seconds,
    )
    summary = run_benchmark(config)
    print(
        "saved benchmark summary to "
        f"{config.output_dir / 'summary.json'}; "
        f"{len(summary['failures'])} failure(s), "
        f"{summary['paired_measurements']['total_paired_n']} fully passed measured pair(s)"
    )
    return 0 if not summary["failures"] else 1


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded local-task-agent Direct/Adaptive benchmark."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--cohort-wall-seconds", type=float, default=40 * 60)
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.cohort_wall_seconds) or arguments.cohort_wall_seconds <= 0:
        parser.error("--cohort-wall-seconds must be finite and greater than zero")
    return arguments


def build_schedule() -> list[dict[str, Any]]:
    """Return the fixed discovery, warmup, and 30-pair measured schedule."""

    schedule: list[dict[str, Any]] = []

    def add(phase: str, pair_index: int, cases: Iterable[str], strategies: Iterable[str]) -> None:
        case_items = tuple(cases)
        strategy_items = tuple(strategies)
        for case_order, case_id in enumerate(case_items, start=1):
            for strategy_order, strategy in enumerate(strategy_items, start=1):
                schedule.append(
                    {
                        "phase": phase,
                        "pair_index": pair_index,
                        "case_id": case_id,
                        "requested_strategy": strategy,
                        "case_order": case_order,
                        "strategy_order": strategy_order,
                        "schedule_index": len(schedule) + 1,
                    }
                )

    # One discovery sample and one separate warmup per requested-strategy/case cell.
    add("discovery", 0, CASES, STRATEGIES)
    add("warmup", 0, CASES, STRATEGIES)
    for pair_index in range(1, MEASURED_PAIRS_PER_CASE + 1):
        if pair_index % 2:
            add("measured", pair_index, CASES, STRATEGIES)
        else:
            add("measured", pair_index, reversed(CASES), reversed(STRATEGIES))
    return schedule


def run_benchmark(
    config: BenchmarkConfig,
    *,
    schedule: Sequence[dict[str, Any]] | None = None,
    process_runner: Callable[[Sequence[str], Path, float], ProcessResult] | None = None,
    identity_reader: Callable[[BenchmarkConfig], dict[str, Any]] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute every scheduled child, retain evidence, and write one summary.

    Injectable collaborators make timeout and failed-child behavior testable
    without a model server or a subprocess.
    """

    planned = build_schedule() if schedule is None else list(schedule)
    ensure_new_output_directory(config.output_dir)
    process_runner = process_runner or run_child_process
    identity_reader = identity_reader or capture_invocation_identity
    started = monotonic()
    deadline = started + config.cohort_wall_seconds
    environment: Any | None = None
    records: list[dict[str, Any]] = []
    initial_failure: str | None = None

    try:
        environment = load_json(config.environment_json)
        frozen_identity = identity_reader(config)
        if frozen_identity["source_dirty"]:
            initial_failure = "source worktree is dirty; benchmark requires a clean frozen source"
    except Exception as error:  # preflight failures must still produce an auditable summary.
        frozen_identity = None
        initial_failure = f"preflight failed: {error}"

    if initial_failure:
        records.extend(
            not_started_record(slot, initial_failure, frozen_identity) for slot in planned
        )
        return write_summary(config, planned, records, frozen_identity, environment, started, monotonic())

    assert frozen_identity is not None
    for index, slot in enumerate(planned):
        remaining = deadline - monotonic()
        if remaining <= 0:
            reason = "cohort wall deadline elapsed before this invocation"
            records.append(not_started_record(slot, reason, frozen_identity))
            records.extend(
                not_started_record(later, reason, frozen_identity)
                for later in planned[index + 1 :]
            )
            break

        try:
            current_identity = identity_reader(config)
        except Exception as error:
            reason = f"could not capture invocation identity: {error}"
            records.append(not_started_record(slot, reason, None))
            records.extend(
                not_started_record(later, reason, None) for later in planned[index + 1 :]
            )
            break
        if current_identity != frozen_identity:
            reason = "executable, source, suite, or environment identity drifted after freeze"
            records.append(not_started_record(slot, reason, current_identity))
            records.extend(
                not_started_record(later, reason, current_identity)
                for later in planned[index + 1 :]
            )
            break

        remaining = deadline - monotonic()
        if remaining <= 0:
            reason = "cohort wall deadline elapsed while capturing invocation identity"
            records.append(not_started_record(slot, reason, current_identity))
            records.extend(
                not_started_record(later, reason, current_identity)
                for later in planned[index + 1 :]
            )
            break

        record = run_one_invocation(
            config,
            slot,
            environment,
            current_identity,
            process_runner,
            max(0.001, remaining),
            monotonic,
        )
        records.append(record)
        print(
            "benchmark "
            f"phase={slot['phase']} case={slot['case_id']} "
            f"strategy={slot['requested_strategy']} "
            f"passed={record['fully_passed']}",
            file=sys.stderr,
            flush=True,
        )
        if slot["phase"] in {"discovery", "warmup"} and not record["fully_passed"]:
            reason = (
                f"{slot['phase']} admission failed at {record['planned_key']}; "
                "measured execution was not admitted"
            )
            records.extend(
                not_started_record(later, reason, frozen_identity)
                for later in planned[index + 1 :]
            )
            break

    return write_summary(config, planned, records, frozen_identity, environment, started, monotonic())


def run_one_invocation(
    config: BenchmarkConfig,
    slot: dict[str, Any],
    environment: Any,
    identity: dict[str, Any],
    process_runner: Callable[[Sequence[str], Path, float], ProcessResult],
    timeout_seconds: float,
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    invocation_dir = invocation_directory(config.output_dir, slot)
    invocation_dir.mkdir(parents=True, exist_ok=False)
    report_path = invocation_dir / "report.json"
    command = build_cli_command(config, slot, report_path)
    started = monotonic()
    try:
        process = process_runner(command, config.source_root, timeout_seconds)
    except Exception as error:
        process = ProcessResult(None, "", "", error=f"could not start child: {error}")
    outer_elapsed_ms = round((monotonic() - started) * 1000, 3)
    stdout_path = invocation_dir / "stdout.txt"
    stderr_path = invocation_dir / "stderr.txt"
    write_new_text(stdout_path, process.stdout)
    write_new_text(stderr_path, process.stderr)

    record = {
        **slot,
        "planned_key": planned_key(slot),
        "invocation_identity": identity,
        "command": list(command),
        "child_timeout_seconds": timeout_seconds,
        "outer_subprocess_elapsed_ms": outer_elapsed_ms,
        "execution_state": "timed_out" if process.timed_out else "completed",
        "exit_code": process.returncode,
        "process_error": process.error,
        "stdout": artifact_reference(stdout_path, config.output_dir),
        "stderr": artifact_reference(stderr_path, config.output_dir),
        "raw_report": None,
        "actual_strategy": None,
        "fallbacks": [],
        "metrics": empty_metrics(),
        "fully_passed": False,
        "failure_reasons": [],
    }
    if process.timed_out:
        record["failure_reasons"].append("child timed out at remaining cohort deadline")
    if process.error:
        record["failure_reasons"].append(process.error)
    if process.returncode not in (0, None):
        record["failure_reasons"].append(f"child exited with status {process.returncode}")
    if not report_path.is_file():
        record["failure_reasons"].append("child produced no raw evaluation artifact")
        return record

    record["raw_report"] = artifact_reference(report_path, config.output_dir)
    try:
        artifact = load_json(report_path)
    except Exception as error:
        record["failure_reasons"].append(f"raw evaluation artifact was not valid JSON: {error}")
        return record
    artifact_errors, extracted = validate_and_extract_artifact(
        artifact, slot, config, identity, environment
    )
    record.update(extracted)
    record["failure_reasons"].extend(artifact_errors)
    record["fully_passed"] = not record["failure_reasons"]
    return record


def build_cli_command(
    config: BenchmarkConfig, slot: dict[str, Any], report_path: Path
) -> list[str]:
    """Build the public live-evaluator invocation without a shell."""

    return [
        str(config.binary),
        "--model",
        config.model,
        "--ollama-url",
        config.ollama_url,
        "--suite",
        str(config.suite),
        "--case",
        str(slot["case_id"]),
        "--repeat",
        "1",
        "--strategy",
        str(slot["requested_strategy"]),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--output-tokens",
        "2048",
        "--max-model-calls",
        "9",
        "--max-tool-calls",
        "8",
        "--max-model-call-duration-ms",
        "120000",
        "--max-run-duration-ms",
        "300000",
        "--environment-json",
        str(config.environment_json),
        "--output",
        str(report_path),
    ]


def run_child_process(command: Sequence[str], cwd: Path, timeout_seconds: float) -> ProcessResult:
    """Run one child with a bounded deadline and retain both output streams."""

    try:
        child = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return ProcessResult(None, "", "", error=f"could not start child: {error}")
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
        return ProcessResult(child.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            stdout, stderr = child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
        return ProcessResult(child.returncode, stdout, stderr, timed_out=True)


def capture_invocation_identity(config: BenchmarkConfig) -> dict[str, Any]:
    """Capture all frozen inputs immediately before a child invocation."""

    if not config.binary.is_file():
        raise FileNotFoundError(f"evaluation binary does not exist: {config.binary}")
    if not config.suite.is_file():
        raise FileNotFoundError(f"benchmark suite does not exist: {config.suite}")
    if not config.environment_json.is_file():
        raise FileNotFoundError(
            f"environment metadata does not exist: {config.environment_json}"
        )
    commit = git_output(config.source_root, ["rev-parse", "HEAD"])
    status = git_output(config.source_root, ["status", "--porcelain"])
    return {
        "executable_sha256": sha256_file(config.binary),
        "source_commit": commit.strip(),
        "source_dirty": bool(status),
        "source_status_sha256": sha256_text(status),
        "suite_sha256": sha256_file(config.suite),
        "environment_sha256": sha256_file(config.environment_json),
    }


def git_output(source_root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        shell=False,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git returned a nonzero status")
    return result.stdout


def validate_and_extract_artifact(
    artifact: Any,
    slot: dict[str, Any],
    config: BenchmarkConfig,
    identity: dict[str, Any],
    expected_environment: Any,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    extracted = {
        "actual_strategy": None,
        "fallbacks": [],
        "metrics": empty_metrics(),
    }
    if not isinstance(artifact, dict):
        return ["raw evaluation artifact root was not an object"], extracted
    if artifact.get("format_version") != 1:
        errors.append("raw evaluation artifact format_version was not 1")
    invocation = artifact.get("invocation")
    if not isinstance(invocation, dict):
        return errors + ["raw evaluation artifact lacked invocation metadata"], extracted
    if invocation.get("source_commit") != identity["source_commit"]:
        errors.append("raw artifact source commit did not match pre-invocation identity")
    if invocation.get("source_dirty") is not False:
        errors.append("raw artifact did not retain a clean source invocation")
    if invocation.get("ollama_base_url") != config.ollama_url:
        errors.append("raw artifact Ollama URL did not match the scheduled loopback URL")
    if invocation.get("selected_models") != [config.model]:
        errors.append("raw artifact model selection did not match the scheduled model")
    if invocation.get("environment") != expected_environment:
        errors.append("raw artifact environment metadata did not match the scheduled input")
    if invocation.get("generation") != {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 2048,
    }:
        errors.append("raw artifact generation settings did not match the fixed benchmark settings")
    if invocation.get("limits") != {
        "max_model_calls": 9,
        "max_tool_calls": 8,
        "max_run_duration_ms": 300000,
        "max_model_call_duration_ms": 120000,
    }:
        errors.append("raw artifact resource limits did not match the fixed benchmark settings")

    evaluation = artifact.get("evaluation")
    if not isinstance(evaluation, dict):
        return errors + ["raw evaluation artifact lacked evaluation data"], extracted
    report = evaluation.get("report")
    evidence = evaluation.get("evidence")
    results = report.get("results") if isinstance(report, dict) else None
    if not isinstance(results, list) or len(results) != 1:
        return errors + ["raw evaluation artifact did not contain exactly one result"], extracted
    if not isinstance(evidence, list) or len(evidence) != 1:
        return errors + ["raw evaluation artifact did not contain exactly one evidence sample"], extracted
    result = results[0]
    sample = evidence[0]
    if not isinstance(result, dict) or not isinstance(sample, dict):
        return errors + ["raw result or evidence sample was not an object"], extracted
    for label, value in (("result", result), ("evidence", sample)):
        if value.get("suite_id") != BENCHMARK_SUITE_ID:
            errors.append(f"raw {label} suite ID did not match the benchmark suite")
        if value.get("case_id") != slot["case_id"]:
            errors.append(f"raw {label} case ID did not match the schedule")
        if value.get("model") != config.model:
            errors.append(f"raw {label} model did not match the schedule")
        if value.get("repetition") != 1:
            errors.append(f"raw {label} repetition was not the required child-local value 1")
    if result.get("strategy") != slot["requested_strategy"]:
        errors.append("raw result requested strategy did not match the schedule")
    if sample.get("requested_strategy") != slot["requested_strategy"]:
        errors.append("raw evidence requested strategy did not match the schedule")
    if result.get("passed") is not True or result.get("failures"):
        errors.append("strict evaluator result did not pass")

    strategy = sample.get("strategy")
    if not isinstance(strategy, dict):
        errors.append("raw evidence lacked strategy evidence")
        strategy = {}
    extracted["actual_strategy"] = strategy.get("actual")
    extracted["fallbacks"] = strategy.get("fallbacks", [])
    metrics = extract_metrics(result, sample)
    extracted["metrics"] = metrics
    errors.extend(effect_sanity_errors(slot["case_id"], sample))
    return errors, extracted


def effect_sanity_errors(case_id: str, sample: dict[str, Any]) -> list[str]:
    """Check only cross-cutting audit/effect boundaries; the Rust contract owns the oracle."""

    errors: list[str] = []
    executions = sample.get("tool_executions")
    approvals = sample.get("approvals")
    if not isinstance(executions, list) or not isinstance(approvals, list):
        return ["raw evidence lacked tool-execution or approval evidence"]
    if sample.get("audit_violations"):
        errors.append("raw evidence recorded an audit violation")
    tool_ids = {
        execution.get("context", {}).get("tool_id")
        for execution in executions
        if isinstance(execution, dict)
    }
    if case_id == "independent-reads-8":
        if tool_ids != {"get_task"} or len(executions) != 8:
            errors.append("eight-read benchmark had an unexpected dispatch boundary")
        if approvals:
            errors.append("eight-read benchmark crossed an approval boundary")
        if sample.get("initial_state") != sample.get("final_state"):
            errors.append("eight-read benchmark changed the task store")
    elif case_id == "dependent-lookup-update":
        if not tool_ids.issubset({"list_tasks", "update_task"}):
            errors.append("dependent control had an unexpected dispatch boundary")
        if any(
            approval.get("context", {}).get("tool_id") != "update_task"
            for approval in approvals
            if isinstance(approval, dict)
        ):
            errors.append("dependent control had an unexpected approval boundary")
    else:
        errors.append("schedule contained an unsupported benchmark case")
    return errors


def extract_metrics(result: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    model_calls = sample.get("model_calls")
    executions = sample.get("tool_executions")
    if not isinstance(model_calls, list):
        model_calls = []
    if not isinstance(executions, list):
        executions = []
    input_tokens = 0
    output_tokens = 0
    tool_attempts = 0
    for call in model_calls:
        if not isinstance(call, dict):
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if isinstance(usage, dict):
            input_tokens += integer_or_zero(usage.get("input_tokens"))
            output_tokens += integer_or_zero(usage.get("output_tokens"))
        calls = response.get("tool_calls")
        if isinstance(calls, list):
            tool_attempts += len(calls)
    mutations = sum(
        1
        for execution in executions
        if isinstance(execution, dict)
        and execution.get("context", {}).get("tool_id") not in {"list_tasks", "get_task"}
    )
    violations = sample.get("audit_violations")
    return {
        "evaluator_duration_ms": result.get("duration_ms"),
        "model_calls": result.get("model_calls"),
        "model_call_evidence": len(model_calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_attempts": tool_attempts,
        "tool_dispatches": len(executions),
        "audit_violations": len(violations) if isinstance(violations, list) else None,
        "mutation_dispatches": mutations,
    }


def write_summary(
    config: BenchmarkConfig,
    planned: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    frozen_identity: dict[str, Any] | None,
    environment: Any,
    started: float,
    finished: float,
) -> dict[str, Any]:
    failures = [record for record in records if not record.get("fully_passed")]
    completed = [
        record
        for record in records
        if record.get("execution_state") in {"completed", "timed_out"}
    ]
    missing = [record for record in records if record.get("execution_state") == "not_started"]
    measured_records = [record for record in records if record.get("phase") == "measured"]
    summary = {
        "format_version": 1,
        "failures": [failure_view(record) for record in failures],
        "all_planned": {
            "count": len(planned),
            "rows": [planned_key(slot) for slot in planned],
        },
        "completed": {
            "count": len(completed),
            "rows": [record["planned_key"] for record in completed],
        },
        "missing": {
            "count": len(missing),
            "rows": [failure_view(record) for record in missing],
        },
        "cohort": {
            "model": config.model,
            "ollama_url": config.ollama_url,
            "cohort_wall_seconds": config.cohort_wall_seconds,
            "outer_cohort_elapsed_ms": round((finished - started) * 1000, 3),
            "frozen_identity": frozen_identity,
            "environment": environment,
        },
        "totals": totals(measured_records),
        "per_case": per_case_summary(measured_records),
        "all_phase_totals": totals(records),
        "phases": phase_summaries(planned, records),
        "paired_measurements": paired_measurements(records),
        "records": list(records),
    }
    write_new_json(config.output_dir / "summary.json", summary)
    return summary


def paired_measurements(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_case_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("phase") == "measured":
            by_case_pair[(record["case_id"], record["pair_index"])][
                record["requested_strategy"]
            ] = record
    per_case: dict[str, Any] = {}
    total_paired_n = 0
    for case_id in CASES:
        comparisons: dict[str, list[float]] = defaultdict(list)
        excluded: list[int] = []
        total_pairs = 0
        for pair_index in range(1, MEASURED_PAIRS_PER_CASE + 1):
            total_pairs += 1
            pair = by_case_pair.get((case_id, pair_index), {})
            direct = pair.get("direct")
            adaptive = pair.get("adaptive")
            if not (direct and adaptive and direct.get("fully_passed") and adaptive.get("fully_passed")):
                excluded.append(pair_index)
                continue
            for metric in (
                "evaluator_duration_ms",
                "model_calls",
                "input_tokens",
                "output_tokens",
                "tool_attempts",
                "tool_dispatches",
                "mutation_dispatches",
            ):
                direct_value = direct["metrics"].get(metric)
                adaptive_value = adaptive["metrics"].get(metric)
                if isinstance(direct_value, (int, float)) and isinstance(
                    adaptive_value, (int, float)
                ):
                    comparisons[metric].append(adaptive_value - direct_value)
        paired_n = total_pairs - len(excluded)
        total_paired_n += paired_n
        per_case[case_id] = {
            "total_pairs": total_pairs,
            "paired_n": paired_n,
            "excluded_pair_count": len(excluded),
            "excluded_pair_indexes": excluded,
            "median_adaptive_minus_direct": {
                metric: median(values) for metric, values in comparisons.items()
            },
            "latency_metric": "evaluator_duration_ms",
            "outer_subprocess_elapsed_ms_is_not_used_for_paired_latency": True,
        }
    return {"total_paired_n": total_paired_n, "per_case": per_case}


def phase_summaries(
    planned: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for phase in ("discovery", "warmup", "measured"):
        phase_planned = [slot for slot in planned if slot.get("phase") == phase]
        phase_records = [record for record in records if record.get("phase") == phase]
        summaries[phase] = {
            "planned": len(phase_planned),
            "completed": sum(
                record.get("execution_state") in {"completed", "timed_out"}
                for record in phase_records
            ),
            "missing": sum(
                record.get("execution_state") == "not_started" for record in phase_records
            ),
            "fully_passed": sum(record.get("fully_passed", False) for record in phase_records),
            "totals": totals(phase_records),
            "per_case": per_case_summary(phase_records),
        }
    return summaries


def per_case_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case_id in CASES:
        case_records = [record for record in records if record.get("case_id") == case_id]
        requested = Counter(record.get("requested_strategy") for record in case_records)
        actual = Counter(
            record.get("actual_strategy")
            for record in case_records
            if record.get("actual_strategy") is not None
        )
        fallbacks = Counter(
            json.dumps(record.get("fallbacks", []), sort_keys=True)
            for record in case_records
            if record.get("fallbacks")
        )
        output[case_id] = {
            "requested_strategy_counts": dict(sorted(requested.items())),
            "actual_strategy_counts": dict(sorted(actual.items())),
            "fallback_sequences": dict(sorted(fallbacks.items())),
            "fully_passed": sum(record.get("fully_passed", False) for record in case_records),
            "metrics": totals(case_records),
        }
    return output


def totals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = Counter()
    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric in (
            "model_calls",
            "model_call_evidence",
            "input_tokens",
            "output_tokens",
            "tool_attempts",
            "tool_dispatches",
            "audit_violations",
            "mutation_dispatches",
        ):
            value = metrics.get(metric)
            if isinstance(value, int):
                values[metric] += value
    values["rows"] = len(records)
    values["fully_passed_rows"] = sum(record.get("fully_passed", False) for record in records)
    return dict(values)


def not_started_record(
    slot: dict[str, Any], reason: str, identity: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        **slot,
        "planned_key": planned_key(slot),
        "invocation_identity": identity,
        "command": None,
        "child_timeout_seconds": None,
        "outer_subprocess_elapsed_ms": None,
        "execution_state": "not_started",
        "exit_code": None,
        "process_error": None,
        "stdout": None,
        "stderr": None,
        "raw_report": None,
        "actual_strategy": None,
        "fallbacks": [],
        "metrics": empty_metrics(),
        "fully_passed": False,
        "failure_reasons": [reason],
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "evaluator_duration_ms": None,
        "model_calls": None,
        "model_call_evidence": None,
        "input_tokens": None,
        "output_tokens": None,
        "tool_attempts": None,
        "tool_dispatches": None,
        "audit_violations": None,
        "mutation_dispatches": None,
    }


def planned_key(slot: dict[str, Any]) -> str:
    return ":".join(
        (
            str(slot["phase"]),
            str(slot["case_id"]),
            str(slot["pair_index"]),
            str(slot["requested_strategy"]),
        )
    )


def invocation_directory(output_dir: Path, slot: dict[str, Any]) -> Path:
    return output_dir / str(slot["phase"]) / (
        f"{int(slot['schedule_index']):03d}-{slot['case_id']}-{slot['requested_strategy']}"
    )


def artifact_reference(path: Path, output_dir: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": sha256_file(path),
    }


def failure_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "planned_key": record["planned_key"],
        "execution_state": record["execution_state"],
        "exit_code": record["exit_code"],
        "failure_reasons": record["failure_reasons"],
        "raw_report": record["raw_report"],
    }


def ensure_new_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory must not already exist: {path}")
    path.mkdir(parents=True, exist_ok=False)


def write_new_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        output.write(text)


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        json.dump(value, output, indent=2, sort_keys=False)
        output.write("\n")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def integer_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


if __name__ == "__main__":
    sys.exit(main())
