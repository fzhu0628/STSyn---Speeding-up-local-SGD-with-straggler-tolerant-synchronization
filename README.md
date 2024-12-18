# STSyn---Speeding-up-local-SGD-with-straggler-tolerant-synchronization
## Overview
- This is my second journal paper titled _Speeding up local SGD with straggler-tolerant synchronization_.
- This work was published at **IEEE Transactions on Signal Processing** (TSP) in 2024.
- About this paper:
  - This work considers the distributed/federated learning framework with _synchronous local SGD_.
  - It is dedicated to improving the robustness of the federated system to potential **stragglers**.
  - To this end, we propose a novel local SGD strategy called **_STSyn_**.
  - The **key idea** is to wait for the $K$ fastest workers while **keeping all the workers computing continually** at each synchronization round, and **making full use of any effective (completed) local update** of each worker regardless of stragglers.
