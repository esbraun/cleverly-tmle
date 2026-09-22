"""The RM18 runtime isolation: code state crossed with runtime on two longitudinal studies.

``docs/roadmap.md`` declares the diagnostic, its preconditions and its reading rule under RM18,
in "The runtime isolation, declared before it runs".  :mod:`.row_drift` reads the committed
history alone, and :mod:`.compare` reads the four scratch arms against it.
"""
