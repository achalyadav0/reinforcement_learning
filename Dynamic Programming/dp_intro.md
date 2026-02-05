# What is Dynamic Programming?

Dynamic Programming (DP) is a computational method used to optimize **multi-stage** (or sequential) decision processes.

It breaks a complex problem into smaller subproblems and solves them in a way that builds up to the optimal overall solution.

---

# Bellman’s Principle of Optimality

**Statement:**

If you are following the best overall plan, then no matter where you currently are, the rest of your plan must also be the best from that point onward.

This means an optimal solution is made up of optimal sub-solutions.

---

# Markov Property

The **Markov Property** states:

> The future depends only on the present state, not on how you reached it.

In other words, once we know the current state, past history is irrelevant for making optimal future decisions.

---

## Mathematical Form (in Reinforcement Learning)

\[
P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots) = P(S_{t+1} \mid S_t, A_t)
\]

This equation shows that the next state depends only on the **current state** and **current action**, not on earlier states or actions.