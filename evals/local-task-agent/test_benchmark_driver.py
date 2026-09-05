import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import benchmark_driver as driver


IDENTITY = {
    "executable_sha256": "binary",
    "source_commit": "3a0edd7",
    "source_dirty": False,
    "source_status_sha256": "clean",
    "suite_sha256": "suite",
    "environment_sha256": "environment",
}


def slot(phase="discovery", case_id="independent-reads-8", strategy="direct", index=1):
    return {
        "phase": phase,
        "pair_index": 0 if phase != "measured" else 1,
        "case_id": case_id,
        "requested_strategy": strategy,
        "case_order": 1,
        "strategy_order": 1,
        "schedule_index": index,
    }


def passing_artifact(slot_data, config, environment):
    executions = [
        {"context": {"tool_id": "get_task"}} for _ in range(8)
    ]
    return {
        "format_version": 1,
        "invocation": {
            "source_commit": IDENTITY["source_commit"],
            "source_dirty": False,
            "ollama_base_url": config.ollama_url,
            "selected_models": [config.model],
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 2048,
            },
            "limits": {
                "max_model_calls": 9,
                "max_tool_calls": 8,
                "max_run_duration_ms": 300000,
                "max_model_call_duration_ms": 120000,
            },
            "environment": environment,
        },
        "evaluation": {
            "report": {
                "results": [
                    {
                        "suite_id": driver.BENCHMARK_SUITE_ID,
                        "case_id": slot_data["case_id"],
                        "model": config.model,
                        "strategy": slot_data["requested_strategy"],
                        "repetition": 1,
                        "passed": True,
                        "failures": [],
                        "duration_ms": 100,
                        "model_calls": 2,
                        "tool_calls": 8,
                    }
                ]
            },
            "evidence": [
                {
                    "suite_id": driver.BENCHMARK_SUITE_ID,
                    "case_id": slot_data["case_id"],
                    "model": config.model,
                    "requested_strategy": slot_data["requested_strategy"],
                    "repetition": 1,
                    "initial_state": {"tasks": []},
                    "final_state": {"tasks": []},
                    "strategy": {"actual": "direct", "fallbacks": []},
                    "model_calls": [
                        {
                            "response": {
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                                "tool_calls": [{}, {}, {}, {}, {}, {}, {}, {}],
                            }
                        },
                        {
                            "response": {
                                "usage": {"input_tokens": 4, "output_tokens": 3},
                                "tool_calls": [],
                            }
                        },
                    ],
                    "tool_executions": executions,
                    "approvals": [],
                    "audit_violations": [],
                }
            ],
        },
    }


class BenchmarkDriverTests(unittest.TestCase):
    def test_wall_budget_rejects_nonfinite_and_nonpositive_values(self):
        common = [
            "--binary", "binary",
            "--suite", "suite",
            "--environment-json", "environment",
            "--source-root", "source",
            "--output-dir", "output",
            "--model", "model",
            "--ollama-url", "http://[::1]:11434",
            "--cohort-wall-seconds",
        ]
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    driver.parse_arguments([*common, value])

    def test_fixed_schedule_is_complete_unique_and_balanced(self):
        schedule = driver.build_schedule()
        self.assertEqual(len(schedule), 128)
        self.assertEqual(len({driver.planned_key(row) for row in schedule}), 128)
        measured = [row for row in schedule if row["phase"] == "measured"]
        self.assertEqual(len(measured), 120)
        for case_id in driver.CASES:
            rows = [row for row in measured if row["case_id"] == case_id]
            self.assertEqual(len(rows), 60)
            self.assertEqual({row["pair_index"] for row in rows}, set(range(1, 31)))
            self.assertEqual(
                sum(row["strategy_order"] == 1 and row["requested_strategy"] == "direct" for row in rows),
                15,
            )
            self.assertEqual(
                sum(row["strategy_order"] == 1 and row["requested_strategy"] == "adaptive" for row in rows),
                15,
            )
        self.assertEqual(
            len({
                row["pair_index"]
                for row in measured
                if row["case_order"] == 1 and row["case_id"] == driver.CASES[0]
            }),
            15,
        )
        for pair_index in range(1, 31):
            pair = [row for row in measured if row["pair_index"] == pair_index]
            self.assertEqual(len(pair), 4)
            self.assertEqual(
                [row["case_id"] for row in pair],
                ([driver.CASES[0], driver.CASES[0], driver.CASES[1], driver.CASES[1]]
                 if pair_index % 2 else
                 [driver.CASES[1], driver.CASES[1], driver.CASES[0], driver.CASES[0]]),
            )
            self.assertEqual(
                [row["requested_strategy"] for row in pair],
                (["direct", "adaptive", "direct", "adaptive"]
                 if pair_index % 2 else
                 ["adaptive", "direct", "adaptive", "direct"]),
            )

    def test_paired_reducer_excludes_failed_and_missing_rows_from_latency(self):
        def row(case_id, pair_index, strategy, passed, duration):
            return {
                "phase": "measured",
                "case_id": case_id,
                "pair_index": pair_index,
                "requested_strategy": strategy,
                "fully_passed": passed,
                "metrics": {
                    "evaluator_duration_ms": duration,
                    "model_calls": 2,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "tool_attempts": 8,
                    "tool_dispatches": 8,
                    "mutation_dispatches": 0,
                },
            }

        records = [
            row("independent-reads-8", 1, "direct", True, 100),
            row("independent-reads-8", 1, "adaptive", True, 130),
            row("independent-reads-8", 2, "direct", True, 100),
            row("independent-reads-8", 2, "adaptive", False, 1),
        ]
        summary = driver.paired_measurements(records)["per_case"]["independent-reads-8"]
        self.assertEqual(summary["paired_n"], 1)
        self.assertIn(2, summary["excluded_pair_indexes"])
        self.assertEqual(
            summary["median_adaptive_minus_direct"]["evaluator_duration_ms"], 30
        )

    def test_failed_subprocess_missing_artifact_and_timeout_fail_closed(self):
        scenarios = {
            "subprocess_failure": driver.ProcessResult(7, "out", "err"),
            "missing_artifact": driver.ProcessResult(0, "out", "err"),
            "timeout": driver.ProcessResult(None, "out", "err", timed_out=True),
        }
        for name, result in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                environment_path = root / "environment.json"
                environment_path.write_text('{"host":"test"}', encoding="utf-8")
                config = driver.BenchmarkConfig(
                    binary=root / "binary.exe",
                    suite=root / "suite.yaml",
                    environment_json=environment_path,
                    source_root=root,
                    output_dir=root / "results",
                    model="gemma4:e4b-it-q4_K_M",
                    ollama_url="http://[::1]:11434",
                )
                schedule = [slot("discovery", index=1), slot("measured", index=2)]
                with contextlib.redirect_stderr(io.StringIO()):
                    summary = driver.run_benchmark(
                        config,
                        schedule=schedule,
                        identity_reader=lambda _: IDENTITY,
                        process_runner=lambda *_: result,
                    )
                self.assertTrue(summary["failures"])
                self.assertFalse(summary["records"][0]["fully_passed"])
                self.assertIn("child produced no raw evaluation artifact", summary["records"][0]["failure_reasons"])
                self.assertEqual(summary["records"][1]["execution_state"], "not_started")
                self.assertTrue((config.output_dir / "summary.json").is_file())
                if name == "subprocess_failure":
                    self.assertIn("child exited with status 7", summary["records"][0]["failure_reasons"])
                if name == "timeout":
                    self.assertIn("child timed out at remaining cohort deadline", summary["records"][0]["failure_reasons"])

    def test_reducer_rejects_a_mismatched_raw_case_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_path = root / "environment.json"
            environment = {"host": "test"}
            environment_path.write_text(json.dumps(environment), encoding="utf-8")
            config = driver.BenchmarkConfig(
                binary=root / "binary.exe",
                suite=root / "suite.yaml",
                environment_json=environment_path,
                source_root=root,
                output_dir=root / "results",
                model="gemma4:e4b-it-q4_K_M",
                ollama_url="http://[::1]:11434",
            )
            schedule = [slot()]

            def write_wrong_case(command, *_):
                output = Path(command[command.index("--output") + 1])
                artifact = passing_artifact(schedule[0], config, environment)
                artifact["evaluation"]["report"]["results"][0]["case_id"] = "wrong-case"
                output.write_text(json.dumps(artifact), encoding="utf-8")
                return driver.ProcessResult(0, "ok", "")

            with contextlib.redirect_stderr(io.StringIO()):
                summary = driver.run_benchmark(
                    config,
                    schedule=schedule,
                    identity_reader=lambda _: IDENTITY,
                    process_runner=write_wrong_case,
                )
            self.assertFalse(summary["records"][0]["fully_passed"])
            self.assertIn(
                "raw result case ID did not match the schedule",
                summary["records"][0]["failure_reasons"],
            )

    def test_identity_capture_time_counts_against_the_child_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_path = root / "environment.json"
            environment_path.write_text("{}", encoding="utf-8")
            config = driver.BenchmarkConfig(
                binary=root / "binary.exe",
                suite=root / "suite.yaml",
                environment_json=environment_path,
                source_root=root,
                output_dir=root / "results",
                model="gemma4:e4b-it-q4_K_M",
                ollama_url="http://[::1]:11434",
                cohort_wall_seconds=1,
            )
            timestamps = iter((0.0, 0.0, 2.0, 2.0))
            summary = driver.run_benchmark(
                config,
                schedule=[slot()],
                identity_reader=lambda _: IDENTITY,
                process_runner=lambda *_: self.fail("deadline-expired child must not start"),
                monotonic=lambda: next(timestamps),
            )
            self.assertEqual(summary["records"][0]["execution_state"], "not_started")
            self.assertIn(
                "cohort wall deadline elapsed while capturing invocation identity",
                summary["records"][0]["failure_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
