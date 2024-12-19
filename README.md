# STSyn: Speeding Up Local SGD with Straggler-Tolerant Synchronization

## Overview

- This is my second journal paper titled _Speeding Up Local SGD with Straggler-Tolerant Synchronization_.
- The paper was published in **IEEE Transactions on Signal Processing (TSP)** in 2024.
- **About this paper**:
  - Focuses on distributed/federated learning with _synchronous local SGD_.
  - Aims to improve **robustness** of federated systems against **stragglers**.
  - Proposes a novel local SGD strategy, **_STSyn_**, with the following key features:
    - Waits for the $K$ fastest workers while ensuring **continuous computation** for all workers.
    - Utilizes **all effective (completed) local updates** from every worker, even with stragglers.
  - Provides rigorous convergence rates for _nonconvex_ objectives under both _homogeneous_ and _heterogeneous_ data distributions.
  - Validates the algorithm through simulations and investigates the impact of system hyperparameters.

## Key Ideas of STSyn

- The system consists of $M$ workers, each performing $U$ local updates per round.
- The server waits for the $K$-th fastest worker to finish $U$ updates.  
- **Key Concept**: No worker stops computing until the $K$-th fastest one completes $U$ updates.  
- Workers that have completed **at least one update** upload their models to the server for aggregation.

### Algorithm Illustration

![image](https://github.com/user-attachments/assets/7200d0f8-7506-41d2-a271-24444ab4a79c)

- Example: $M=4$, $U=3$, $K=3$.  
  - Workers 1, 2, and 3 are the fastest $K=3$ workers to complete $U=3$ updates in round 0.
  - **Red arrows**: Additional updates performed by the fastest $K-1=2$ workers.  
  - **Light blue arrows**: Straggling updates that are cancelled.  
  - All 4 workers upload their models, as each completes at least one update.

### Pseudocode

Below is the pseudocode for the STSyn algorithm:

![image](https://github.com/user-attachments/assets/30b9e0a1-0ecb-4d5d-8a4a-ab4d6c412582)

## Analysis

### Average Wall-Clock Time and Number of Local Updates and Uploading Workers

- Assuming the time for a single local update by each worker follows an exponential distribution, we provide closed-form expressions for:
  - The average wall-clock time per round.
  - The average number of local updates per worker per round.
  - The average number of uploading workers per round.

### Convergence Analysis

- **Heterogeneous Data Distributions**:
  - The average expected squared gradient norm for nonconvex objectives is upper bounded by:
    
    $$O\left(\frac{1}{\sqrt{K\bar{U} J}} + \frac{K}{\bar{U}^3 J}\right)$$

    where:
    - $K$: Number of workers the server waits for.
    - $\bar{U}$: Average local updates per worker per round.
    - $J$: Total number of communication rounds.

- **Homogeneous Data Distributions**:
  - The convergence rate is the same as above:
    
    $$O\left(\frac{1}{\sqrt{K\bar{U} J}} + \frac{K}{\bar{U}^3 J}\right)$$


## Simulation Results

![image](https://github.com/user-attachments/assets/b7ab638e-323a-4903-a249-c35583e1c198)

- Numerical experiments validate that the STSyn algorithm achieves superior time and communication efficiency under both i.i.d. and non-i.i.d. data distributions among workers.

For more details, refer to the full paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10659740).

## Codes

- **`Comparison.py`**:
  - Compares STSyn with state-of-the-art algorithms on the CIFAR-10 dataset using a three-layer CNN.
- **`impact_K`**:
  - Explores the impact of the hyperparameter $K$.
- **`impact_U`**:
  - Explores the impact of the hyperparameter $U$.
