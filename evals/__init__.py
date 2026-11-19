"""Evaluation harness for AgentDesk.

W1: `python -m evals.smoke` runs 5 canned tasks through the hand-rolled loop with
a scripted fake client — proving the eval pipeline (run → record → assert) is
alive in CI with zero API cost. W5 replaces this with the real task sets, user
simulator, LLM-as-Judge and Pass^k measurement (book Ch7 methodology).
"""
