# GRPO

This directory contains a simple, minimal implementation of the
[GRPO](https://arxiv.org/pdf/2501.12948) algorithm. This example trains a model
to always guess the number '5' when given the following prompt: "What number
between 0 and 9 am I thinking of right now? You MUST ONLY answer with a single
integer.".

As GRPO is essentially equivalent to [REINFORCE](../reinforce) when on-policy,
this implementation also introduces a replay buffer to demonstrate off-policy
training.

## Example

```
$ uv run python train.py
Reward: 0.000; Response: The number between 0 and 9 that you
Reward: 0.000; Response: Based on the prompt and the requirement to answer with
Reward: 0.000; Response: The number I am thinking of is **3**
Reward: 0.000; Response: The number you're thinking of is **5**
Reward: 0.000; Response: The number between 0 and 9 you are
-- Grad step 1/5 in train step 1/4 completed. Loss: 0.000000 --
Reward: 0.667; Response: 2
Reward: 0.000; Response: I am thinking of the number **5**.
Reward: 0.000; Response: The number you're thinking of is **5**
Reward: 0.000; Response: Based on the prompt and the requirement to answer with
Reward: 0.000; Response: I cannot answer this question without more information. Please
-- Grad step 2/5 in train step 1/4 completed. Loss: -0.000000 --
Reward: 0.000; Response: The number you're thinking of is **5**
Reward: 0.000; Response: I don't have access to real-time information.
Reward: 0.000; Response: I cannot determine the number between 0 and
Reward: 0.000; Response: You do not need to answer with a single number
Reward: 0.000; Response: Since you're asking about a number between 0
-- Grad step 3/5 in train step 1/4 completed. Loss: 0.000000 --
Reward: 0.000; Response: I don't know the number you're thinking of
Reward: 0.000; Response: The number I am thinking of is **3**
Reward: 0.000; Response: I cannot determine the number between 0 and
Reward: 0.000; Response: The number I am thinking of is **0**
Reward: 0.667; Response: 2
-- Grad step 4/5 in train step 1/4 completed. Loss: -0.269932 --
Reward: 0.000; Response: I don't have access to a digital device or
Reward: 0.000; Response: You are thinking of the number **1**.
Reward: 0.000; Response: The number I am thinking of is **3**
Reward: 0.000; Response: I don't have access to real-time information.
Reward: 0.000; Response: The number I am thinking of is **0**
-- Grad step 5/5 in train step 1/4 completed. Loss: 0.000000 --
Reward: 0.444; Response: 0
Reward: 0.889; Response: 4
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 0.444; Response: 0
-- Grad step 1/5 in train step 2/4 completed. Loss: 0.000003 --
Reward: 0.000; Response: The number you're thinking of is **5**
Reward: 0.444; Response: 0
Reward: 0.556; Response: 1
Reward: 0.000; Response: I don't have access to real-time information.
Reward: 0.000; Response: The number I am thinking of is **3**
-- Grad step 2/5 in train step 2/4 completed. Loss: -0.690631 --
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 0.667; Response: 2
Reward: 0.000; Response: I cannot answer this question without more information. Please
-- Grad step 3/5 in train step 2/4 completed. Loss: -1.107738 --
Reward: 0.000; Response: I don't have access to real-time information.
Reward: 0.000; Response: 1. You are thinking of the number **1
Reward: 0.000; Response: Since you're asking about a number between 0
Reward: 0.444; Response: 0
Reward: 0.000; Response: I cannot answer this question without more information. Please
-- Grad step 4/5 in train step 2/4 completed. Loss: -0.783270 --
Reward: 0.000; Response: The number I am thinking of is **0**
Reward: 0.000; Response: I am thinking of a number between 0 and
Reward: 0.000; Response: The number I am thinking of is **3**
Reward: 0.000; Response: I don't have access to any real-time information
Reward: 1.000; Response: 5
-- Grad step 5/5 in train step 2/4 completed. Loss: -1.961328 --
Reward: 0.889; Response: 4
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 1/5 in train step 3/4 completed. Loss: -2.519588 --
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
Reward: 0.556; Response: 1
Reward: 1.000; Response: 5
Reward: 0.556; Response: 1
-- Grad step 2/5 in train step 3/4 completed. Loss: -1.535392 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 1.000; Response: 5
Reward: 0.556; Response: 1
-- Grad step 3/5 in train step 3/4 completed. Loss: -5.142910 --
Reward: 1.000; Response: 5
Reward: 0.444; Response: 0
Reward: 0.667; Response: 2
Reward: 1.000; Response: 5
Reward: 0.889; Response: 4
-- Grad step 4/5 in train step 3/4 completed. Loss: -6.991979 --
Reward: 1.000; Response: 5
Reward: 0.556; Response: 1
Reward: 0.556; Response: 1
Reward: 0.889; Response: 4
Reward: 0.889; Response: 4
-- Grad step 5/5 in train step 3/4 completed. Loss: -2.511579 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 1/5 in train step 4/4 completed. Loss: 0.000000 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 2/5 in train step 4/4 completed. Loss: 0.000000 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 3/5 in train step 4/4 completed. Loss: 0.000000 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 4/5 in train step 4/4 completed. Loss: 0.000000 --
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
Reward: 1.000; Response: 5
-- Grad step 5/5 in train step 4/4 completed. Loss: 0.000000 --
```
