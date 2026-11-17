# 461-Program-2-Genetic-Algorithms

## 1. Initialize Population

Randomly generate 250 schedules.
Each schedule is a chromosome containing 11 genes (activities).
Each gene contains three values: Room, Time, and Facilitator.

2. Calculate Fitness
Evaluate each schedule using the fitness rules provided in Appendix A.
Award or deduct points based on room capacity, scheduling conflicts, facilitator preferences, workload, and special SLA101/SLA191 rules.
Calculate the Best, Average, and Worst Fitness for each generation.

3. Parent Selection (Softmax)
Convert fitness scores into selection probabilities using Softmax.
Schedules with higher fitness have a greater chance of being selected as parents.
Lower-fitness schedules can still be selected to maintain diversity.

4. Crossover
Randomly select two parents.
Perform Single-Point Crossover by exchanging parts of their chromosomes.
Each crossover produces two children.

5. Mutation
Apply random mutations to the children after crossover.
Each room, time, or facilitator variable has a probability of being replaced with another valid value.
The initial mutation rate is 0.01 (1%).

6. Adaptive Mutation Rate
After each generation, compare its Average Fitness with the previous generation:
Improvement ≥ 1%: Halve the mutation rate for the next generation.
Improvement < 1% before Generation 100: Keep the mutation rate unchanged and continue.
Improvement < 1% at or after Generation 100: Stop the algorithm.
The mutation rate is adjusted automatically during execution.

7. Generate the Next Generation
Repeat parent selection, crossover, and mutation until 250 children are generated.
These children completely replace the previous population.
Repeat the process until the stopping condition is satisfied.

8. Output and Evaluation
The main program generates:
best_schedule.txt – Best schedule from the final generation, including fitness breakdown and constraint diagnostics.
fitness_history.csv – Fitness statistics and mutation rate changes for each generation.
fitness_plot.png – Best, Average, and Worst Fitness over generations.
mutation_rate_plot.png – Adaptive mutation rate changes over generations.

