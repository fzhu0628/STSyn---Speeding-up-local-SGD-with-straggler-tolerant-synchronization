# STSyn---Speeding-up-local-SGD-with-straggler-tolerant-synchronization
## Overview
- This is my second journal paper titled _Speeding up local SGD with straggler-tolerant synchronization_.
- This work was published at **IEEE Transactions on Signal Processing** (TSP) in 2024.
- About this paper:
  - This work considers the distributed/federated learning framework with _synchronous local SGD_.
  - It is dedicated to improving the **robustness** of the federated system to potential **stragglers**.
  - To this end, we propose a novel local SGD strategy called **_STSyn_**.
  - The **key idea** is to wait for the $K$ fastest workers while **keeping all the workers computing continually** at each synchronization round, and **making full use of any effective (completed) local update** of each worker regardless of stragglers.
  - Rigorous convergence rates of STSyn are provided with _nonconvex_ objectives, both for _homogeneous_ and _heteregeneous_ data distributions.
  - Simulation results are provided to validate the superiority of our algorithm.
  - Impact of system hyper-parameters is also investigated.
## Illustration of STSyn

  ![image](https://github.com/user-attachments/assets/7200d0f8-7506-41d2-a271-24444ab4a79c)
  
In this figure, we assume a total of 4 workers. We ask each one of them to perform 3 local updates and the server waits for the third fast worker to at least complete 3 local updates. As illustrated, the red arrows represent the additional local updates performed by the fastest 2 workers, and the arrows in light blue represent the straggling updates that are cancelled.
## Analysis

