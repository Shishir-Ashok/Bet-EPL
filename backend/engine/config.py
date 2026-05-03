# The DQN only executes a bet if Q(action) - Q(PASS) >= this value.
# Must match what the agent was trained with (train.py MIN_BET_CONFIDENCE).
DQN_CONFIDENCE_GATE = 0.5