# STSyn---Speeding-Up-Local-SGD-with-Straggler-Tolerant-Synchronization
## Overview
- This is my second journal paper titled _Speeding up local SGD with straggler-tolerant synchronization_.
- This work was published at **IEEE Transactions on Signal Processing** (TSP) in 2024.
- **About this paper**
  - This work considers the distributed/federated learning framework with _synchronous local SGD_.
  - It is dedicated to improving the **robustness** of the federated system to potential **stragglers**.
  - To this end, we propose a novel local SGD strategy called **_STSyn_**.
  - The **key idea** is to wait for the $K$ fastest workers while **keeping all the workers computing continually** at each synchronization round, and **making full use of any effective (completed) local update** of each worker regardless of stragglers.
  - Rigorous convergence rates of STSyn are provided with _nonconvex_ objectives, both for _homogeneous_ and _heteregeneous_ data distributions.
  - Simulation results are provided to validate the superiority of our algorithm.
  - Impact of system hyper-parameters is also investigated.
## Key Ideas of STSyn
Consider a total of $M$ workers, we ask each worker to perform $U$ local updates, and the server waits for the $K$-th fastest worker that have finished $U$ local updates. Workers that have finished **at least 1 update** are required to upload their updated models to the server for aggregation.

  ![image](https://github.com/user-attachments/assets/7200d0f8-7506-41d2-a271-24444ab4a79c)
  
Illustration of STSyn with $M=4$, $U=3$ and $K=3$. In words, we ask each worker to perform 3 local updates, and the server only waits for the third fastest worker to have completed the task. In round 0, workers 1, 2, and 3 are the fastest $K=3$ workers that have completed $U=3$ local updates; the red arrows represent the additional local updates performed by the fastest $K-1=2$ workers; and the arrows in light blue represent the straggling updates that are cancelled. **All four workers are required to upload their models** since they have completed at least 1 local update.
## Analysis
### Analysis of Average Wall-Clock Time and Number of Local Updates and Uploading Workers
We assume that the wall-clock time of each worker computing one local update in each round conforms to exponential distribution, and provide closed-form expression for:
- the average wall-clock runtime taken per round
- the average number of local updates taken by each worker per round
- the average number of uploading workers per round.
### Convergence Analysis
Under smoothness, lower boundedness, unbiasedness and variance boundedness, we prove that
- For workers with different local loss functions, the average of expected sum of squared norm of gradients with nonconvex objectives is upper bounded by $O\left(\frac{1}{\sqrt{K\bar U J}}+\frac{K}{\bar U^3J}\right)$ where
  - $K$ is the number of workers the server waits for.
  - $\bar U$ is the average number of local updates performed by each worker in each round.
  - J is the total number of communication rounds.
- For workers with same local loss functions, the average of expected sum of squared norm of gradients with nonconvex objectives is upper bounded by $O\left(\frac{1}{\sqrt{K\bar U J}}+\frac{K}{\bar U^3J}\right)$, **same as the above one**.
## Simulation Results

