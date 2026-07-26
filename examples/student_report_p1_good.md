 t# Task summary
I created paper cards for recent work on LLM-driven point-cloud and 3D scene reasoning. The work maps each paper by problem, method, data, metric, claim, limitation, and how dangerous it is to our project.

# Parent research question and claim
This supports C1: the topic can be decomposed into a defensible roadmap after prior-art and gap analysis. It also supports E1: literature map, closest prior-art list, gap matrix, and reviewer-objection memo.

# Method or procedure
I searched for 3D-LLM, point-cloud LLM, spatial reasoning, benchmark leakage, 3D QA, and tool-use terms. For each selected paper, I extracted problem, method, dataset, metric, stated contribution, limitation, and relation to our candidate direction.

# Output artifacts
- paper_cards.md
- dangerous_prior_art_matrix.csv
- gap_cards.md
- reviewer_objection_notes.md

# Result interpretation
The strongest publishable direction is not generic 3D QA. The better direction is evidence-grounded point-cloud reasoning with explicit geometry tools, answer evidence IDs, and checks against 2D or language-only shortcuts.

# Limitations and confounders
The survey is still incomplete because it needs a second pass on the newest 2025-2026 papers and benchmark-specific failure analyses. Some papers may use different terminology, so synonym search is needed.

# Reproducibility commands
No code execution was required. Search queries and paper list are recorded in paper_cards.md.

# Next step recommendation
PI should review the dangerous prior-art matrix and approve which candidate contribution angle should become the benchmark/method roadmap.
