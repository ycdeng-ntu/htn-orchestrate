"""CLI entry points for running benchmarks.

- ``run_one_task``    runs one question in this Python subprocess.
- ``run_subprocess``  fans out into many ``run_one_task`` subprocs.
- ``check_progress``  monitors / aggregates accuracy of an in-flight run.

Each is invocable via ``python -m benchmarks.runner.<name>``.
"""
