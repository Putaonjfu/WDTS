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

*(Note: All code for the complete WDTS pipeline, including both skeletonization and interwoven optimization, is now fully released and available.)*

**Testing with Sample Data:**
We provide a `testdata` folder containing sample data to help you quickly test the pipeline. The folder includes:
* `Tree_1.txt`: The raw point cloud data.
* `Tree_1_ske_without_interwoven_op.ply`: The initial skeletonization result (for reference or as input for Step 2).
* `Tree_1_ske_within_interwoven_op.ply`: The final skeleton after interwoven optimization (for reference).

### Step 1: Initial Skeletonization

First, configure the input data path, output save path, tree ID, and gamma value directly inside the `run_skeletonization.py` script. You can point these to your data to run the code:

```python
# Please set the path to your dataset, the base save path, and the ID of the tree to be skeletonized.
# Note: The input file should be in txt format, but do not include the extension ('.txt').
datapath = '/home/graper/WDTS_test/data/'
base_savepath = '/home/graper/WDTS_test/result_test/'
Name = ['Tree_16']

# Set the gamma value for water droplet contraction and merging. We recommend 0.1.
gamma_values = [0.1]
```

After modifying the paths, run the skeletonization script to generate the initial individual tree skeleton:

```bash
python run_skeletonization.py
```

### Step 2: Interwoven Optimization

Next, apply the geometric and topological interwoven optimization to the preliminary skeleton.

Configure the input and output directories directly inside the `run_interwoven_optimization.py` script. You can point these to your skeleton and TLS directories:

```python
# Please set the input directory of the skeleton files for interwoven optimization.
self.sk_input_dir = r"./testdata/"

# Please set the input directory of the corresponding point cloud files.
self.tls_input_dir = r"./testdata/"

# Please set the output directory for the optimization results.
self.output_dir = r"./testdata/result_test/"
```

After modifying the paths to match your local setup, simply run:

```bash
python run_interwoven_optimization.py
```

**Note on Downstream Applications:**
Please note that the code for precise individual tree modeling and structural parameter extraction is currently not publicly available, as it is part of another ongoing research project. Please watch or star this repository to stay tuned for future updates!

---

## Citation

If you find our work, methodology, or code useful in your research, please consider citing our paper.

**Note:** The manuscript is currently under **major revision** at the *ISPRS Journal of Photogrammetry and Remote Sensing*. The citation will be updated once accepted.

```bibtex
@article{wdts_major_revision,
  title={WDTS: Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds},
  author={Tao Pu...},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={Major Revision}
}
```

---

## Star History

## Star History

<a href="https://www.star-history.com/?repos=Putaonjfu/WDTS&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=Putaonjfu/WDTS&type=date&theme=dark&legend=top-left" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=Putaonjfu/WDTS&type=date&legend=top-left" />
    <img alt="Star History Chart" src="https://api.star-history.com/image?repos=Putaonjfu/WDTS&type=date&legend=top-left" />
  </picture>
</a>
