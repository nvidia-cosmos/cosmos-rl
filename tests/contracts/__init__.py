# Cosmos-rl plugin-surface contract tests.
#
# These tests assert the implicit contracts that the controller and
# RLPolicyWorker rely on when integrating user-supplied trainers,
# rollout backends, models, and configs. The contracts are encoded
# elsewhere in the codebase (e.g. dict-key reads in
# ``cosmos_rl/dispatcher/status.py`` and attribute reads in
# ``cosmos_rl/policy/worker/rl_worker.py``) but never declared as a
# ``Protocol`` or ABC; these tests pin them down so plugin authors get
# a fast structural check instead of debugging the launcher.
