# FFGT: A Hybrid Focal and Full-Range Attention Based Graph Transformer

FFGT is a purely attention based framework which combines the conventional full-range attention with k-hop focal attention on ego-nets to aggregate both global and local information.

Here is the guidance of running ffgt. All experiments are run on a laptop with AMD Ryzen 9 5950X 16-Core processor, 64.0 GB RAM and one Nvidia RTX3090ti GPU.

**Note**: There are problems with downloading `SBM-PATTERN-NEW datasets` in versions before Aug.27, 2023. Please use the latest version of ffgt to run experiments on these datasets.

### Python environment setup with Conda

```bash
conda create -n ffgt python=3.9.15

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu116.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu116.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.1+cu116.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu116.html
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric==2.1.0

pip install ogb==1.3.5
pip install rdkit
pip install tensorflow==2.11.0
pip install tensorboardx
pip install tqdm

conda --clean all
```

### Running FFGT

```bash
conda activate ffgt

# Running vanilla_ffgt and tuned hyperparameters for ZINC.
python gpp.py --config configs/vanilla/vanilla_ffgt_ZINC.json 

# Running grpe_ffgt and tuned hyperparameters for ZINC.
python gpp.py --config configs/grpe/grpe_ffgt_ZINC.json 

# To run fully connected version, one can set "max_dist" to 0 in config file, or use the command line bellow
python gpp.py --config configs/grpe/grpe_ffgt_ZINC.json --max_dist 0

# change "max_dist" for different focal length
```

## Benchmarking FFGT on 4 empirical datasets
Use `run_gpp.sh` to run multiple random seeds per each of the datasets. Allows for one dataset at a time. Command lines are as follows:

```bash
conda activate ffgt

# Run 4 repeats with 4 different random seeds (0..3) on ZINC:
python gpp.py configs/vanilla/vanilla_ffgt_ZINC.json 0 4

```

## SBM-PATTERN-NEW

Four Synthetic Datasets based on SBM-PATTERN is used to test whether optimal focal length varies with the scale of local substructures. The four datasets are `PATTERN_010_001_005`, `PATTERN_012_001_005`, `PATTERN_014_001_005`, `PATTERN_016_001_005`. They share the same inter community probability $q=0.01$ and inter comunity-pattern probability $q_p=0.05$, while has intra community probability $p$ and intra pattern probability $p_p$ the same: $p=p_p \in \{0.10, 0.12, 0.14, 0.16 \}$ 

### Statistics

| Probability | #Graphs | Avg.#nodes | Avg.#degrees| Avg.#diameters   |
|-------|-------|-------|------|------|
| $p=0.16$ | 12000 | 125 | 6.13 | 6.15  |
| $p=0.14$ | 12000 | 127 | 5.72 | 6.38  |
| $p=0.12$ | 12000 | 129 | 5.34 | 6.60  |
| $p=0.11$ | 12000 | 130 | 4.85 | 7.00  |


### Running SBM-PATTERN-NEW datasets

The four datasets can be downloaded [here](https://github.com/minhongzhu/sbm-pattern-new). To run Vanilla-FFGT on these datasets, one should follow the guidlines as follows:

```bash

    # run PATTERN_010_001_005
    python gpp.py --config configs/vanilla/vanilla_ffgt_PATTERN.json --dataset PATTERN_pq_010_001_005 --out_dir out/SBM-PATTERN/pq_012_001_005/vanilla/

    # or you can modify the --dataset and --out_dir direcly in "configs/vanilla/vanilla_ffgt_PATTERN.json" and run 
    python gpp.py --config configs/vanilla/vanilla_ffgt_PATTERN.json

```

Note that raw files (.zip) of SBM-PATTERN-NEW datasets should be put under `./dataset/SBM-PATTERN/` in FFGT Directory.
