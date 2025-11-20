# DAPO

This directory contains a simple, minimal implementation of the
[DAPO](https://arxiv.org/pdf/2503.14476) algorithm. This example trains a model
to always guess the number '5' when given the following prompt: "What number
between 0 and 9 am I thinking of right now? You MUST ONLY answer with a single
integer.".

## Dynamic sampling

The DAPO paper introduces 'dynamic sampling' in a slightly ambiguous manner.
This is my interpretation.

Dynamic sampling ensures that your group always gives you a useful training
signal by selecting prompts that are neither "too easy" (rollout rewards are
always 1) or "too hard" (rollout rewards are always 0).

In the implementation they describe (both textually in section 3.2 and via
Algorithm 1), they introduce a 'dynamic sampling buffer' which you populate
only with reward-diverse prompt responses. E.g., if the policy completed a
prompt ten times and every response had a reward of 1, we discard that prompt
(and its responses) altogether. Once the dynamic sampling buffer reaches a
critical size, we would then iterate through groups in the buffer (in batches)
and train on them by maximising the DAPO objective.

This implementation has only a single prompt, so we just continually resample
responses rather than picking new prompts.

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
Dynamic sampling: rejecting group with no reward diversity.
Reward: 0.000; Response: Let me think...
Reward: 0.444; Response: 0
Reward: 0.000; Response: Based on the prompt
Reward: 0.000; Response: The number you are
Reward: 0.000; Response: The number you're
Reward: 0.000; Response: I don't have
Dynamic sampling: accepting group.
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: I am thinking of
Reward: 0.000; Response: I cannot answer this
Reward: 0.000; Response: I cannot ask you
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: You are thinking of
Dynamic sampling: rejecting group with no reward diversity.
Reward: 0.667; Response: 2
Reward: 0.000; Response: The number between
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The answer is:
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The number you're
Dynamic sampling: accepting group.
Reward: 0.667; Response: 2
Reward: 0.000; Response: The number is **
Reward: 0.000; Response: You are thinking of
Reward: 0.000; Response: I cannot determine the
Reward: 0.000; Response: You are thinking of
Reward: 0.000; Response: The number between
Dynamic sampling: accepting group.
Reward: 0.000; Response: I cannot know the
Reward: 0.000; Response: The number you're
Reward: 0.000; Response: The number you're
Reward: 0.444; Response: 0
Reward: 0.000; Response: Based on the given
Reward: 0.000; Response: You cannot determine the
Dynamic sampling: accepting group.
Reward: 0.000; Response: You are thinking of
Reward: 0.000; Response: Based on the information
Reward: 0.444; Response: 0
Reward: 0.000; Response: I do not have
Reward: 0.000; Response: I cannot answer this
Reward: 0.000; Response: I am thinking of
Dynamic sampling: accepting group.
Reward: 0.000; Response: I am not able
Reward: 0.000; Response: You do not need
Reward: 0.000; Response: You must only answer
Reward: 0.000; Response: I don't have
Reward: 0.000; Response: You are thinking of
Reward: 0.000; Response: I am not able
Dynamic sampling: rejecting group with no reward diversity.
Reward: 0.000; Response: The number between
Reward: 0.000; Response: You must answer with
Reward: 0.000; Response: I cannot determine the
Reward: 0.667; Response: 2
Reward: 0.000; Response: The number you're
Reward: 0.000; Response: Based on the information
Dynamic sampling: accepting group.
-- Training
Grad step for group batch in train step 1/4 completed. Batch objective: -0.371135. 4/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 1/4 completed. Batch objective: -0.371136. 2/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 1/4 completed. Batch objective: -0.314749. 0/6 groups remain in dynamic sampling buffer.
-- Generating rollouts
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The number you're
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 0.556; Response: 1
Reward: 0.556; Response: 1
Dynamic sampling: accepting group.
Reward: 0.444; Response: 0
Reward: 0.000; Response: The number I am
Reward: 0.000; Response: The number is
Reward: 0.778; Response: 3
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Dynamic sampling: accepting group.
Reward: 0.667; Response: 2
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.556; Response: 1
Reward: 0.889; Response: 4
Reward: 0.000; Response: The number you're
Dynamic sampling: accepting group.
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.556; Response: 1
Reward: 0.000; Response: The number I am
Reward: 0.667; Response: 8
Dynamic sampling: accepting group.
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.000; Response: The number I am
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
Dynamic sampling: accepting group.
Reward: 0.444; Response: 0
Reward: 0.000; Response: The number I am
Reward: 0.667; Response: 2
Reward: 0.556; Response: 1
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
-- Training
Grad step for group batch in train step 2/4 completed. Batch objective: -0.447866. 4/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 2/4 completed. Batch objective: -0.424746. 2/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 2/4 completed. Batch objective: -0.848991. 0/6 groups remain in dynamic sampling buffer.
-- Generating rollouts
Reward: 0.778; Response: 3
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 0.444; Response: 0
Reward: 0.889; Response: 4
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 0.556; Response: 1
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Dynamic sampling: accepting group.
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.444; Response: 0
Reward: 0.556; Response: 1
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 0.444; Response: 0
Reward: 0.889; Response: 4
Reward: 0.556; Response: 1
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 0.889; Response: 6
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.667; Response: 2
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.778; Response: 3
Reward: 0.889; Response: 4
Dynamic sampling: accepting group.
-- Training
Grad step for group batch in train step 3/4 completed. Batch objective: 0.000007. 4/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 3/4 completed. Batch objective: 0.169887. 2/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 3/4 completed. Batch objective: 0.145898. 0/6 groups remain in dynamic sampling buffer.
-- Generating rollouts
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 0.778; Response: 3
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.889; Response: 6
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Dynamic sampling: accepting group.
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Dynamic sampling: rejecting group with no reward diversity.
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
Dynamic sampling: accepting group.
-- Training
Grad step for group batch in train step 4/4 completed. Batch objective: 0.000013. 4/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 4/4 completed. Batch objective: 0.128880. 2/6 groups remain in dynamic sampling buffer.
Grad step for group batch in train step 4/4 completed. Batch objective: 0.143060. 0/6 groups remain in dynamic sampling buffer.
```
