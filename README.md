# WDTS: Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds

## Abstract

The construction of one-dimensional (1D), centered individual tree curve skeletons that preserve local branch details is a fundamental challenge in forestry remote sensing. Such skeletons are a critical prerequisite for multi-scale studies, ranging from the precise quantification of individual tree parameters to global carbon cycle modeling. To this end, this paper proposes the water droplet model-driven entropy optimization for individual tree skeletonization (WDTS) from Terrestrial Laser Scanning (TLS) point clouds. WDTS models an individual tree TLS point cloud as a system of water droplets with varying masses. The water droplets undergo a process of contraction, coalescence, and evaporation. This evolution is driven by an equivalent surface tension, which is defined by attractive, repulsive, and resultant forces. During this evolution, a normalized skeleton entropy is defined based on the geometric feature distribution of the evolving water droplet model to measure its orderliness. Through an entropy reduction optimization, the disordered water droplet model gradually converges to a low entropy, ordered, 1D curve skeleton that preserves local branch details. Finally, a geometric and topological co-optimization is performed. The method achieves this through an alternating, iterative process. It first minimizes the normalized distance variance from the individual tree TLS point cloud to the nearest skeleton edges and then reconstructs the skeleton topology. This mutual reinforcement between points and edges generates a more centered individual tree curve skeleton. Experiments conducted on a comprehensive dataset of 34 real-world and simulated individual tree TLS point clouds demonstrate the effectiveness of WDTS. Qualitatively, WDTS converges the point cloud into a compact 1D curve skeleton, ensures high fidelity of local details, and exhibits excellent robustness across different data acquisition modes. Quantitatively, the average centeredness accuracy of WDTS ($MAE=0.015m, Sd=0.026m$) is significantly better than that of the second-best method, L1-medial+Dijkstra ($MAE=0.023m, Sd=0.044m$).

---

![Pipeline](Pipieline.png)

---

The code is currently being organized and will be uploaded in the near future.
