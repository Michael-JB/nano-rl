# DAPO

WIP.

## Dynamic sampling

The DAPO paper introduces 'dynamic sampling' in a slightly ambiguous manner.
What follows is my understanding.

Dynamic sampling has two components: the condition in the objective, and the
response sampling strategy. The former ensures that your group always gives you
a useful training signal, and the latter makes this easier by essentially
selecting prompts that are neither "too easy" (rollout rewards are always 1) or
"too hard" (rollout rewards are always 0).

In the implementation they describe (both textually in section 3.2 and via
Algorithm 1), they introduce a 'dynamic sampling buffer' which you populate
only with reward-diverse prompt responses. E.g., if the policy completed a
prompt ten times and every response had a reward of 1, we discard that prompt
(and its responses) altogether. Once the dynamic sampling buffer reaches a
critical size, we would then sample a group from it _such that the group also
has reward diversity_.

This implementation has only a single prompt. As a result, the dynamic sampling
buffer is not applicable, so I stick with a standard replay buffer and only
implement group re-sampling.

## Example

```
$ uv run python train.py
-- Generating rollouts
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: You are thinking of
Reward: 0.000; Response: I cannot determine the
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The number between
Reward: 0.000; Response: I cannot determine the
Reward: 0.000; Response: Let me think...
Reward: 0.444; Response: 0
Reward: 0.000; Response: Based on the prompt
Reward: 0.000; Response: The number you are
Reward: 0.000; Response: The number you're
Reward: 0.000; Response: I don't have
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: I am thinking of
Reward: 0.000; Response: I cannot answer this
Reward: 0.000; Response: I cannot ask you
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: You are thinking of
Reward: 0.667; Response: 2
Reward: 0.000; Response: The number between
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The answer is:
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The number you're
Reward: 0.667; Response: 2
-- Training
Sampled group would give no learning signal; resampling...
Grad step 1/5 in train step 1/6 completed. Objective: -0.198763
Grad step 2/5 in train step 1/6 completed. Objective: -0.198761
Grad step 3/5 in train step 1/6 completed. Objective: -0.165809
Sampled group would give no learning signal; resampling...
Grad step 4/5 in train step 1/6 completed. Objective: -0.170834
Grad step 5/5 in train step 1/6 completed. Objective: -0.242200
-- Generating rollouts
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.889; Response: 4
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
-- Training
Grad step 1/5 in train step 2/6 completed. Objective: -0.192685
Grad step 2/5 in train step 2/6 completed. Objective: -0.201207
Grad step 3/5 in train step 2/6 completed. Objective: -0.176349
Grad step 4/5 in train step 2/6 completed. Objective: -0.313144
Grad step 5/5 in train step 2/6 completed. Objective: -0.181936
-- Generating rollouts
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 0.778; Response: 3
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.778; Response: 3
Reward: 0.667; Response: 2
Reward: 0.889; Response: 4
Reward: 0.778; Response: 3
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.778; Response: 3
Reward: 0.778; Response: 3
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
-- Training
Grad step 1/5 in train step 3/6 completed. Objective: 0.000003
Grad step 2/5 in train step 3/6 completed. Objective: -0.248517
Grad step 3/5 in train step 3/6 completed. Objective: -0.251497
Grad step 4/5 in train step 3/6 completed. Objective: -0.299298
Grad step 5/5 in train step 3/6 completed. Objective: 0.059598
-- Generating rollouts
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 0.778; Response: 3
Reward: 1.000; Response: 5
Reward: 0.778; Response: 3
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.778; Response: 3
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.667; Response: 2
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Training
Grad step 1/5 in train step 4/6 completed. Objective: 0.068380
Grad step 2/5 in train step 4/6 completed. Objective: 0.000697
Grad step 3/5 in train step 4/6 completed. Objective: 0.039447
Grad step 4/5 in train step 4/6 completed. Objective: 0.059183
Grad step 5/5 in train step 4/6 completed. Objective: 0.054716
-- Generating rollouts
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Training
Grad step 1/5 in train step 5/6 completed. Objective: 0.028285
Grad step 2/5 in train step 5/6 completed. Objective: 0.062914
Grad step 3/5 in train step 5/6 completed. Objective: 0.074127
Grad step 4/5 in train step 5/6 completed. Objective: 0.088654
Grad step 5/5 in train step 5/6 completed. Objective: 0.097190
-- Generating rollouts
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
-- Training
Grad step 1/5 in train step 6/6 completed. Objective: 0.038735
Sampled group would give no learning signal; resampling...
Grad step 2/5 in train step 6/6 completed. Objective: 0.074915
Grad step 3/5 in train step 6/6 completed. Objective: 0.075917
Grad step 4/5 in train step 6/6 completed. Objective: 0.076535
Grad step 5/5 in train step 6/6 completed. Objective: 0.067901
```
