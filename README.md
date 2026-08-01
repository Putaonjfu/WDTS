# WDTS

**Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds**

WDTS extracts a compact, centered, and topologically coherent skeleton from an individual-tree TLS point cloud. It combines:

1. Water-droplet skeletonization for the initial skeleton.
2. Geometric-topological interwoven optimization for centeredness and connectivity.

![WDTS pipeline](pipeline.png)

## Quick start

Python 3.10 is recommended.

```bash
conda create -n wdts python=3.10 -y
conda activate wdts
pip install -r requirements.txt
python run_wdts.py pipeline test_data/Tree_1.txt --output results
```

The sample contains 84,556 points and can take tens of minutes on a CPU.

Main outputs:

```text
results/Tree_1/
|-- initial_skeleton.ply
|-- initial_skeleton_points.txt
`-- optimized_skeleton.ply
```

## Commands

Run the complete two-stage pipeline:

```bash
python run_wdts.py pipeline test_data/Tree_1.txt --output results
```

Run skeletonization only:

```bash
python run_wdts.py skeletonize test_data/Tree_1.txt --output results --gamma 0.1
```

Run interwoven optimization only:

```bash
python run_wdts.py optimize --tls test_data/Tree_1.txt --skeleton test_data/Tree_1_ske_without_interwoven_op.ply --output results/Tree_1/optimized_skeleton.ply
```

Use `python run_wdts.py --help` for all options. `python -m wdts` provides the same interface. Add `--save-intermediate` to retain per-iteration diagnostic files.

## Input and output

- TLS input: `.txt`, `.csv`, `.xyz`, `.ply`, or `.pcd` with XYZ in the first three columns. A single `x y z` header is optional.
- Skeleton input for Stage 2: `.ply`, `.txt`, or `.csv`. Stage 2 reconstructs topology from the vertices and does not reuse input PLY edges.
- Input coordinates should use one metric unit and represent one already-segmented tree.
- Output PLY files contain both vertices and indexed edges.

The included `test_data` directory contains the TLS sample and reference skeletons before and after interwoven optimization.

## Python API

```python
from wdts import run_pipeline

result = run_pipeline(
    "test_data/Tree_1.txt",
    output_dir="results",
    gamma=0.1,
)

print(result.optimized_skeleton_path)
```

Stage-specific functions are also available as `run_skeletonization` and `run_interwoven_optimization`. Advanced parameters can be supplied through `SkeletonizationConfig` and `InterwovenOptimizationConfig`.

## Project layout

```text
run_wdts.py                    Unified command-line entry point
wdts/skeletonization.py        Stage 1 algorithm
wdts/interwoven_optimization.py Stage 2 algorithm
wdts/pipeline.py               Two-stage orchestration
test_data/                     Sample input and reference results
tests/                         Fast API and data-contract tests
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The refactor preserves the original algorithm order and numerical formulation while clarifying naming, configuration, I/O, and execution. It also removes hard-coded paths and fixes two debug-version failure paths involving global root state and undefined fallback descriptors.

## Citation

```bibtex
@article{pu2026wdts,
  title={WDTS: Water droplet model-driven entropy optimization for individual tree skeletonization from terrestrial laser scanning point clouds},
  author={Pu, Tao and Du, Shenglan and Sui, Mingming and Chen, Dong and Shen, Yueqian and Chen, Yanming and Kong, Yiyang and Wang, Ziyou and Poovvancheri, Jiju and Zhang, Liqiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={238},
  pages={81--113},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.isprsjprs.2026.04.050}
}
```

## License

See [license](license).
