# WDTS

**Water Droplet Model-Driven Entropy Optimization for Individual Tree Skeletonization from Terrestrial Laser Scanning Point Clouds**

WDTS extracts a compact, centered, and topologically coherent skeleton from an individual-tree TLS point cloud. It combines:

1. Water-droplet skeletonization for the initial skeleton.
2. Geometric-topological interwoven optimization for centeredness and connectivity.

![WDTS pipeline](pipeline.png)

## Reproduction

Python 3.10 is recommended.

```bash
conda create -n wdts python=3.10 -y
conda activate wdts
pip install -r requirements.txt
python run_wdts.py pipeline example_data/Tree_1.txt --output results
```

The sample contains 84,556 points and can take tens of minutes on a CPU.

Main outputs:

```text
results/Tree_1/
|-- initial_skeleton.ply
|-- initial_skeleton_points.txt
`-- optimized_skeleton.ply
```

## Run individual stages

Run skeletonization only:

```bash
python run_wdts.py skeletonize example_data/Tree_1.txt --output results --gamma 0.1
```

Run interwoven optimization only:

```bash
python run_wdts.py optimize --tls example_data/Tree_1.txt --skeleton example_data/Tree_1_ske_without_interwoven_op.ply --output results/Tree_1/optimized_skeleton.ply
```

Use `python run_wdts.py --help` for all options. Add `--save-intermediate` to retain per-iteration diagnostic files.


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
