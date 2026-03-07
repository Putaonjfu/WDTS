# WDTS: Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds

## Abstract

Individual tree skeletonization is a fundamental task in forestry remote sensing, which serves as a crucial prerequisite for various downstream applications, ranging from tree structural parameter estimation to carbon cycle modeling. Nevertheless, most existing skeletonization approaches struggle to generate a compact, centered tree skeleton while preserving detail fidelity and topological rationality. To this end, this paper proposes a water droplet model-driven entropy optimization approach (WDTS) to extract individual tree skeletons from Terrestrial Laser Scanning (TLS) point clouds. WDTS models an individual tree TLS point cloud as a system of water droplets with varying masses, by progressively generating the skeleton through simulated droplet contraction, merging, and evaporation processes. Key to our approach is an entropy reduction framework that progressively drives droplets toward compact skeletons. To further enhance the centeredness of the generated tree skeleton, WDTS employs a geometric and topological interwoven optimization strategy, explicitly aligning the skeleton within the center of the branch point clouds by minimizing the sum of the squared residuals. Experiments conducted on three individual tree TLS point cloud datasets with different data acquisition strategies have demonstrated the effectiveness and robustness of the proposed WDTS. Compared with previous methods, especially the state-of-the-art Dijkstra-enhanced L1-medial method, WDTS remarkably improves the compactness and centeredness of the skeletons with well-preserved local branch details, reducing the averaged MAE by 0.011m, 0.002m, and 0.030m on the single-scan, multi-scans, and simulated dataset, respectively. The generated tree skeletons including not only the tree skeleton points but also topologically coherent edges provide a robust foundation for downstream tasks, including precise tree geometry modeling, biomass estimation, and forestry-related sustainable development applications. 

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
```

---

## Usage

The complete WDTS method consists of a two-step pipeline: initial skeletonization followed by interwoven optimization. 

*(Note: We have currently released the code for the interwoven optimization. The core skeletonization code is being organized and will be uploaded soon.)*

### 1. Full WDTS Pipeline 

Once all modules are available, the end-to-end extraction will be executed in two sequential steps:

**Step 1.1: Skeletonization (Code coming soon)** First, generate the initial individual tree skeleton from the TLS point cloud:
```bash
python run_skeletonization.py --input data/sample_tree.txt --output results/initial_skeleton.ply
```

**Step 1.2: Interwoven Optimization** Next, apply the geometric and topological interwoven optimization to the preliminary skeleton to obtain the final, highly-centered WDTS output:
```bash
python run_interwoven_optimization.py
```

### 2. Running the Available code (Interwoven Optimization)

If you already have a preliminary skeleton (or want to test the interwoven optimization ), you can run it right now. Before running, you must configure the input and output directories directly inside the script. 

Open the relevant Python file and update the following variables with your local absolute paths:

```python
# Please set the input directory of the skeleton files for interwoven optimization.
self.sk_input_dir = r"/path/to/your/skeleton_files_directory"

# Please set the input directory of the corresponding point cloud files.
self.tls_input_dir = r"/path/to/your/point_cloud_directory"

# Please set the output directory for the optimization results.
self.output_dir = r"/path/to/your/output_directory"
```

After modifying the paths to match your local setup, simply run:

```bash
python run_interwoven_optimization.py
```

---

## Citation

If you find our work, methodology, or code useful in your research, please consider citing our paper. 

**Note:** The manuscript is currently **major revision** at the *ISPRS Journal of Photogrammetry and Remote Sensing*. The citation will be updated once accepted.

```bibtex
@article{wdts_under_review,
  title={WDTS: Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds},
  author={Tao Pu...},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={Under Review}
}
```
