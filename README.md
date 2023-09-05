## Aligning Large Language Model with Fair Preference

Ongoing Project

Current Progress: Step 4: fair finetuning


Some Remarks of shapes:

log_probs.shape() = [B,L]

kl_divergence.shape() = [B,L]

reward_score.shape() = [B]

rewards = kl_divergence + reward_score (to the last entry)

rewards.shape() = [B,L]
