# nanl-rl

> Note: this repo is WIP; forgive the messy code, missing docs and lack of TDD.

Playing around with RL as a learning exercise.

## Example

Training a model to guess the number "5":

```
$ uv run python train.py
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
Step 1/10, Loss: 0.270146
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
Step 2/10, Loss: 0.270146
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
Step 3/10, Loss: 0.790735
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
Step 4/10, Loss: 1.046238
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
Step 5/10, Loss: 1.283335
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
Step 6/10, Loss: 1.346148
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
Step 7/10, Loss: 0.972523
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
Step 8/10, Loss: 0.626751
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
Step 9/10, Loss: 0.333392
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
Step 10/10, Loss: 0.318341
```
