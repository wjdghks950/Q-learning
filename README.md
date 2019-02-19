# Q-learning tutorial
This code is a basic Q-learning code.

# Implementation
The implementation of this code is based on examples in the following blog post:

    https://towardsdatascience.com/introduction-to-q-learning-88d1c4f2b49c

# Model
It uses Bellman equation to update the Q-tables, but stops iteration if the past N (i.e. N=10) matrices have the same element-wise value with the current Q-table.

# Run
python qlearning.py