# REINFORCE

This directory contains a simple, minimal implementation of the REINFORCE
algorithm. This example trains a model to always guess the number '5' when given
the following prompt: "What number between 0 and 9 am I thinking of right now?
You MUST ONLY answer with a single integer.".

## Example

```
$ uv run python -m nano_rl.reinforce.train
Reward: 0.0000; Response: The number I am thinking of is **3**
Reward: 0.0000; Response: I don't know the number you're thinking of
Reward: 0.0000; Response: I cannot determine the number between 0 and
Reward: 0.0000; Response: Based on the prompt and the requirement to answer with
Reward: 0.0000; Response: I am thinking of the number **5**.
Reward: 0.0000; Response: I am thinking of a number between 0 and
Reward: 0.0000; Response: The number I am thinking of is **0**
Reward: 0.6667; Response: 2
Reward: 0.0000; Response: The number between 0 and 9 you are
Reward: 0.0000; Response: 1. You are thinking of the number **1
-- Train step 1/10 completed. Objective: -0.270146 --
Reward: 0.6667; Response: 2
Reward: 0.0000; Response: The number is **4**.
Reward: 0.0000; Response: I cannot determine the exact number you're thinking of
Reward: 0.0000; Response: Since you're asking about a number between 0
Reward: 0.0000; Response: The number you're thinking of is **5**
Reward: 0.0000; Response: You cannot determine the number you're thinking of between
Reward: 0.0000; Response: I don't have access to a digital device or
Reward: 0.0000; Response: I don't have access to real-time information.
Reward: 0.0000; Response: You do not need to answer with a single number
Reward: 0.0000; Response: You are thinking of the number **1**.
-- Train step 2/10 completed. Objective: -0.270146 --
Reward: 0.6667; Response: 2
Reward: 0.0000; Response: The number I am thinking of is **2**
Reward: 0.6667; Response: 2
Reward: 0.0000; Response: The number you're thinking of is **0**
Reward: 0.0000; Response: The number I'm thinking of is **3**
Reward: 0.5556; Response: 1
Reward: 0.5556; Response: 1
Reward: 0.0000; Response: The number I am thinking of is 5.
Reward: 0.0000; Response: I don't have access to your thoughts or a
Reward: 0.4444; Response: 0
-- Train step 3/10 completed. Objective: -0.790735 --
Reward: 0.5556; Response: 1
Reward: 0.8889; Response: 4
Reward: 0.0000; Response: The number you're thinking of is 5.
Reward: 0.5556; Response: 1
Reward: 0.0000; Response: The number I am thinking of is 4.
Reward: 0.0000; Response: The number I am thinking of is: **5
Reward: 0.4444; Response: 0
Reward: 0.0000; Response: The number I am thinking of is **3**
Reward: 1.0000; Response: 5
Reward: 0.6667; Response: 2
-- Train step 4/10 completed. Objective: -1.046238 --
Reward: 1.0000; Response: 5
Reward: 0.5556; Response: 1
Reward: 0.4444; Response: 0
Reward: 0.4444; Response: 0
Reward: 0.8889; Response: 4
Reward: 0.6667; Response: 2
Reward: 0.8889; Response: 4
Reward: 0.5556; Response: 1
Reward: 0.6667; Response: 2
Reward: 0.6667; Response: 2
-- Train step 5/10 completed. Objective: -1.283335 --
Reward: 0.8889; Response: 4
Reward: 0.8889; Response: 4
Reward: 0.8889; Response: 4
Reward: 0.4444; Response: 0
Reward: 0.5556; Response: 1
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 0.7778; Response: 3
Reward: 0.5556; Response: 1
-- Train step 6/10 completed. Objective: -1.346148 --
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.6667; Response: 2
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
-- Train step 7/10 completed. Objective: -0.972523 --
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
-- Train step 8/10 completed. Objective: -0.626751 --
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
-- Train step 9/10 completed. Objective: -0.333392 --
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 0.8889; Response: 4
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
Reward: 1.0000; Response: 5
-- Train step 10/10 completed. Objective: -0.318341 --
```
