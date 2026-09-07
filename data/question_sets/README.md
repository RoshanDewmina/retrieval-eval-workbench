# Question-set lineage

`v1.0.0-original-exposed` is the original question-set blob committed at `faee6bf` (`git show faee6bf:data/questions.json`). It is preserved in Git history and is explicitly exposed development history.

`v1.1.0-exposed-development` is the current `../questions.json` file. It changed after the original build, including wording and answer-support adjustments, so it is a regression/development set only. `development-labels-v1.json` adds exact corpus support labels for it.

No file in this directory is a fresh held-out set. Root will independently author and freeze a new set after the correction commit. It must not feed settings selection or implementation changes.
