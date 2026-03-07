# WDTS: Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds

## Abstract

This repository provides the implementation of **WDTS (Water Droplet Model-Driven Entropy Optimization)** for individual tree skeletonization from Terrestrial Laser Scanning (TLS) point clouds.

Individual tree skeletonization is a fundamental task in forestry remote sensing and serves as an important prerequisite for various downstream applications, such as tree structural parameter estimation, tree geometry modeling, biomass estimation, and carbon cycle analysis. However, most existing skeletonization methods still struggle to simultaneously generate a **compact**, **centered**, and **topologically reasonable** skeleton while preserving fine local branch details.

To address this issue, WDTS models an individual tree TLS point cloud as a system of water droplets with varying masses, and progressively generates the skeleton through simulated **droplet contraction, merging, and evaporation** processes. The method is further enhanced by an **entropy reduction framework** and a **geometric and topological interwoven optimization strategy**, which improve skeleton compactness, centeredness, and topological coherence.

---

## Pipeline

![Pipeline](pipeline.png)

---

## Environment Setup

To run WDTS, you should first create a conda environment named `wdts` with Python 3.8, then activate it and install the required dependencies.

```bash
conda create -n wdts python=3.8 -y
conda activate wdts
pip install -r requirements.txt
