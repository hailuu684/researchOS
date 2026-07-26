from __future__ import annotations

from .schemas import StudentLevel, StudentProfile


SAMPLE_STUDENTS = [
    StudentProfile(
        student_id="S_JUNIOR_01",
        name="An",
        level=StudentLevel.junior,
        skills=["paper reading", "data labeling", "running scripts"],
        availability_hours_per_week=6,
        notes="New student. Good fit for structured literature or reproduction tasks.",
    ),
    StudentProfile(
        student_id="S_INTER_01",
        name="Binh",
        level=StudentLevel.intermediate,
        skills=["python", "experiments", "baseline runs", "data processing"],
        availability_hours_per_week=8,
        notes="Can own engineering or experiment packages with review.",
    ),
    StudentProfile(
        student_id="S_STRONG_01",
        name="Chi",
        level=StudentLevel.strong,
        skills=["method design", "benchmark design", "scientific analysis", "writing"],
        availability_hours_per_week=5,
        notes="Can own ambiguous high-criticality packages with PI check-ins.",
    ),
    StudentProfile(
        student_id="S_SENIOR_01",
        name="Dung",
        level=StudentLevel.senior,
        skills=["paper framing", "reviewer response", "project synthesis"],
        availability_hours_per_week=4,
        notes="Best fit for PI-ready synthesis and triage support.",
    ),
]


GOOD_P1_SUBMISSION = """
# Task summary
I built an initial literature map for the PI topic. The output includes paper_cards.md,
dangerous_prior_art_matrix.csv, and gap_cards.md. The goal is to help decide whether the topic
has a defensible research direction before students spend time on implementation.

# Parent research question and claim
This submission supports C1 and E1. The parent research question is: what has already been solved,
which papers are dangerous prior art, and which gap statements are concrete enough for PI review?
The evidence requirement is a literature map, closest prior-art list, gap matrix, and reviewer-objection memo.

# Method or procedure
I separated papers into method papers, benchmark papers, dataset papers, and analysis papers. For each paper card,
I extracted problem, method, data, metric, main claim, limitation, and relation to our topic. I then ranked papers
as dangerous prior art when they overlap with our problem, method, metric, or claimed contribution.

# Output artifacts
- paper_cards.md
- dangerous_prior_art_matrix.csv
- gap_cards.md
- reviewer_objection_notes.md

# Result interpretation
The strongest candidate gap is not simply using LLMs for the topic. The safer angle is to require explicit evidence,
traceable evaluation, fair baselines, and failure analysis. This makes the work more paper-ready because it turns a vague
topic into specific claims and evidence requirements.

# Limitations and confounders
The survey may miss papers using different terminology. The current dangerous prior-art list is not final. The gap can be
overclaimed if we do not compare against simple baselines and near-miss systems. I also have not verified all citations yet.

# Reproducibility commands
No code was run for this task. The reproducibility path is the search log in search_log.md, the seed paper list,
and the extraction template used for every paper card. Version: literature_map_v0.1. Commit: demo-commit-001.

# Next step recommendation
PI or senior student should review the top dangerous prior-art list, then approve or mutate 2-3 gap cards into candidate
research questions for P2.
"""


WEAK_P1_SUBMISSION = """
I read papers and found that our topic is new. I think we can publish if we use LLMs.
The output is some notes. There are no big limitations.
"""


GOOD_UNDERSTANDING_ANSWERS = [
    "A dangerous prior-art paper is one that overlaps with our parent claim, evidence requirement, method, benchmark, metric, or final paper framing. It matters because PI should not approve implementation until we know whether the contribution is actually different from close prior work.",
    "The weakest gap is any gap that only says 'use LLM for this topic' without specifying a testable claim, baseline, metric, or evidence. It is weak because it can be attacked as incremental or solved by an existing method using different terminology.",
    "This package supports C1 and E1 by turning a broad topic into paper cards, a dangerous prior-art matrix, gap cards, and reviewer objections. It helps PI decide whether to approve, mutate, or reject the direction before assigning high-criticality tasks.",
    "The artifact supports the prior-art and novelty gate. The limitation is that the search may miss papers with different keywords, so the evidence is not final until a mentor samples the citations and near-miss list.",
]


WEAK_UNDERSTANDING_ANSWERS = [
    "I think it is useful because I read many papers.",
    "The gap is good and new.",
    "The PI can use it for the project.",
]
