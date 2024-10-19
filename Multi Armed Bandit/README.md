# Multi Armed Bandit (MAB) Problems

## 1. Simple Bandit Algorithm

Algorithm which can solve stationary MAB problem. Action values converged to its true value as steps increase. It found the true greedy action too.

The problem is, the converge speed of non-greedy actions is quite slow. It can be improved by using larger ε value. But larger ε may cause lower average reward. There is a trade-off relation.

```
# Initialize
FOR a = 1 to 5
    q(a) ← 0
    cnt_action($a$) ← 0

LOOP n_step
    # Select an action
    A ← greedy action (with probability 1-ε)
        random action (with probability ε)

    # Do the action
    R ← bandit(A)
    cnt_action(A) ← cnt_action(A) + 1

    # Update
    q(A) ← q(A) + 1 / cnt_action(A) * (R - q(A))
```

<p align="center">
<img src="./1. Simple Bandit Algorithm/plot1_1500steps.png" width="800px">
Average of received rewards (1500 steps)
</p>

<p align="center">
<img src="./1. Simple Bandit Algorithm/plot2_18000steps.png" width="800px">
Action values (18000 steps)
</p>

## 2. Weighted Average

Algorithm which perform well on non-stationary MAB problem. Because the α of the update rule is constant, this algorithm can adapt new environment better than using decaying α (Simple Bandit Algorithm).
<p align="center">
<img src="./2. Weighted Average/plot1_weighted_average.png" width="800px">
Use weighted average (constant α)</br>It finds new greedy actions well
</p>

<p align="center">
<img src="./2. Weighted Average/plot1_non_weighted_average.png" width="800px">
Simple bandit algorithm (decaying α)</br>It cannot find greedy actions of new environments
</p>

## 3. Optimistic Initial Values

## 4. Upper Confidence Bound (UCB)
