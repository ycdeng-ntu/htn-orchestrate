"""Shared JSONL field schema used by every standard family DatasetConfig."""

STD_SCHEMA = dict(
    id_field="task_id",
    question_field="task_question",
    answer_field="ground_truth",
    file_field="file_name",
    file_name_field="file_name",
)
