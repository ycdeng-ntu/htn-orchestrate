"""Pipeline spec: ``dynamic_htn_informed_research``.

A single heavy node (``dynamic_solve``) owns the whole DYNAMIC HTN loop —
lazy per-node decomposition + per-leaf multi-round interactive dispatch. Like
``adaptive_htn`` / ``swarm``, the tree is NOT expressed as DAG edges (MiniDAG
topology is frozen at compile time); all looping lives inside the node.
"""

from __future__ import annotations

from agent_harness.models.pipeline_spec import (
    CompressionConfig,
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)

DYNAMIC_HTN_INFORMED_RESEARCH_SPEC = PipelineSpec(
    pipeline_id="dynamic_htn_informed_research",
    name="Result-Informed Dynamic HTN Orchestration",
    description=(
        "Incremental HTN decomposition informed by bounded completed-precursor "
        "summaries, with a per-leaf multi-round interactive orchestrator."
    ),
    entry_point="dynamic_solve",
    terminal_nodes=["dynamic_solve"],
    nodes=[
        NodeDefinition(
            node_id="dynamic_solve",
            role_id="dynir_researcher",
            node_function=(
                "workflows.dynamic_htn_informed_research.nodes.dynamic_solve.dynamic_solve_node"
            ),
            context_policy=ContextPolicy(
                include_fields=[
                    "original_question",
                    "task_question",
                    "language",
                    "task_id",
                    "metadata",
                ],
            ),
            compression=CompressionConfig(enabled=False),
            output_fields=[
                "report",
                "final_answer",
                "final_content",
                "task_tree",
                "delivery_mode",
                "task_strategy",
                "usage_summary",
                "artifact_dir",
            ],
        ),
    ],
    transitions=[
        TransitionSpec(from_phase="dynamic_solve", to_phase="__END__"),
    ],
)
