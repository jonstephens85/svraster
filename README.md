# Sparse Voxels Rasterizer

![teaser](./asset/teaser.jpg)

### [Project page](https://svraster.github.io) | [Arxiv](https://arxiv.org/abs/2412.04459) | [jupyter notebooks](./notebooks/)

<details>
<summary>Paper abstract</summary>
We propose an efficient radiance field rendering algorithm that incorporates a rasterization process on adaptive sparse voxels without neural networks or 3D Gaussians. There are two key contributions coupled with the proposed system. The first is to adaptively and explicitly allocate sparse voxels to different levels of detail within scenes, faithfully reproducing scene details with 65536^3 grid resolution while achieving high rendering frame rates. Second, we customize a rasterizer for efficient adaptive sparse voxels rendering. We render voxels in the correct depth order by using ray direction-dependent Morton ordering, which avoids the well-known popping artifact found in Gaussian splatting. Our method improves the previous neural-free voxel model by over 4db PSNR and more than 10x FPS speedup, achieving state-of-the-art comparable novel-view synthesis results. Additionally, our voxel representation is seamlessly compatible with grid-based 3D processing techniques such as Volume Fusion, Voxel Pooling, and Marching Cubes, enabling a wide range of future extensions and applications.
</details>

**Updates:**
- Mar 18, 2025: Revise literature review. Support depthanythingv2 relative depth loss and mast3r metric depth loss for a better geometry.
- Mar 8, 2025: Support ScanNet++ dataset. Check the [benchmark](https://kaldir.vc.in.tum.de/scannetpp/benchmark/nvs) for our results on the 3rd-party hidden set evaluation. Our [short article](./articles/scannetpp_dataset.md) may be helpful if you want to work on scannet or indoor environement.

## READ THIS BEFORE INSTALL

I created this fork from the oiginal svraster project to guide users through installing svraster on Windows PCs. This can also be installed on Linux skipping any Windows specific instructions. Please refer to the original project page for the latest developments as this fork does not contain future developments and will not be maintained long term.

## Install
This project has a codebase that is currently only compatible with Linux. However, you can run this project on a Windows machine by setting up Ubuntu via WSL. [I have a complete video guide here](https://youtu.be/xyOdoZp4u_c).

After setting up Ubuntu via WSL, follow these steps:

### Create a conda environment:
```
conda create -n svraster python=3.9
conda activate svraster
```

### Install Pytorch
The tested versions are `1.13.1+cu117` and `2.5.0+cu124`.

#### Option 1: Install PyTorch 2.5.0 with CUDA 12.4
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```
Ensure your environment includes the CUDA 12.4 toolkit (if you have 12.4 installed system-wide, skip this step)
```
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit
```

#### Option 2: Install PyTorch 1.13.1 with CUDA 11.7
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
Also install the matching toolkit:
```
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit
```

### Pull Source Code
```
git clone https://github.com/NVlabs/svraster.git
cd svraster
```

### Install requirements, rasterizer, and utilities
```
pip install -r requirements.txt
pip install -e cuda/
```

## Reconstructing your own capture
Below go through the workflow for reconstruction from a scene capturing. Check [example.ipynb](./notebooks/example.ipynb) for an actual example.

### Data preparation
The original project recommends following [InstantNGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#colmap) video or images processing steps to extract camera parameters using COLMAP. [NerfStudio](https://docs.nerf.studio/quickstart/custom_dataset.html) also works.

We now only support pinhole camera mode. Please preprocess with `--colmap_camera_model PINHOLE` of InstantNGP script or `--camera-type pinhole` of NerfStudio script.

You can use the lightweight script I made following these instructions:

1. Download the script:
[Download run_colmap.sh](https://github.com/jonstephens85/svraster/raw/main/run_colmap.sh)

2. Your image data should be placed in the following structure inside the repository:
```
svraster/
├── run_colmap.sh
├── data/
│   └── my-project/
│       ├── images/              # <-- Place input images here
│       └──                     # database.db and colmap results will be generated here
```

3. From the root of the svraster/ repo, run:
```
chmod +x run_colmap.sh            # Only need to run this the first time using the script
./run_colmap.sh <project-name> [--matcher exhaustive]
```

- <project-name> is the name of the folder inside data/ that contains your images.
- Default matcher is sequential. Use --matcher exhaustive to run exhaustive matching.

4. Exmaple uage:
```
./run_colmap.sh nvidia
```

To override the default matcher and run exhaustive matching:
```
./run_colmap.sh nvidia --matcher exhaustive
```

### Scene optimization
```bash
python train.py --eval --source_path $DATA_PATH --model_path $OUTPUT_PATH
```
All the results will be saved into the specified `$OUTPUT_PATH` including the following results:
- `config.yaml`: The config file for reproduction.
- `pg_view/`: Visualization of the training progress. Useful for debugging.
- `test_stat/`: Some statistic during the training.
- `test_view/`: Some visualization during the training.

The configuration is defined by the following three, the later overwrites the former.
- `src/config.py`: Define the configuable setup and their initial values.
- `--cfg_files`: Sepcify a list of config files, the later overwrites the former. Some examples are under `cfg/`.
- command line: Any field defined in `src/config.py` can be overwritten through command line. For instances: `--data_device cpu`, `--subdivide_save_gpu`.

Like InstantNGP and other NeRF variants, defining a proper main scene bounding box is crucial to quality and processing time. Note that the main scene bound is defined for the main 3D region of interest. There are another `--outside_level` (default 5) Octree levels for the background region. The default main scene bound heuristic may work well in many cases but you can manually tweak them for a better results or covering new type of capturing trajectory:
- `--bound_mode`:
    - `default`
        - Use the suggested bbox if given by dataset. Otherwise, it automatically chose from `forward` or `camera_median` modes.
    - `camera_median`
        - Set camera centroid as world origin. The bbox radius is set to the median distance between origin and cameras.
    - `camera_max`
        - Set camera centroid as world origin. The bbox radius is set to the maximum distance between origin and cameras.
    - `forward`
        - Assume [LLFF](https://github.com/Fyusion/LLFF?tab=readme-ov-file#local-light-field-fusion) forward-facing capturing. See `src/utils/bounding_utils.py` for detail heuristic.
    - `pcd`
        - Use COLMAP sparse points to compute a scene bound. See `src/utils/bounding_utils.py` for detail heuristic.
- `--bound_scale`: scaling the main scene bound (default 1).

For scenes with background masked out, use `--white_background` or `--black_background` to specify the background color.

Other hyperparameter suggestions:
- Ray termination
    - `--lambda_T_inside 0.01` to encourage rays to stop inside the Octree. Useful for real-world scenes.
    - `--lambda_T_concen 0.1` to encourage transmittance to be either 0 or 1. Useful for scenes whose background pixels are set to white or black. Remember to set either `--white_background` or `--black_background` in this case.
- Geometry
    - `--lambda_normal_dmean 0.001 --lambda_normal_dmed 0.001` to encourage self-consistency between rendered depth and normal.
        - Also cite [2dgs](https://arxiv.org/abs/2403.17888) if you use this in your research.
    - `--lambda_ascending 0.01` to encourage density to be increasing along ray direction.
    - `--lambda_sparse_depth 0.01` to use COLMAP sparse points loss to guide rendered depth.
        - Also cite [dsnerf](https://arxiv.org/abs/2107.02791) if you use this in your research.
    - `--lambda_depthanythingv2 0.1` to use depthanythingv2 loss to guide rendered depth.
        - It uses the huggingface version.
        - It automatically saves the estimated depth map at the first time you activate this loss for the scene.
        - Also cite [depthanythingv2](https://arxiv.org/abs/2406.09414) and [midas](https://arxiv.org/abs/1907.01341) if you use this in your research.
    - `--lambda_mast3r_metric_depth 0.1` to use the metric depth derived from MASt3R to guide the rendered depth.
        - You need to clone MASt3R and install all its dependency.
        - You also need to set `--mast3r_repo_path {abs_path_to_mast3r_repo}`.
        - It automatically saves the estimated depth map at the first time you activate this loss for the scene.
        - Also cite [MASt3R](https://arxiv.org/abs/2406.09756) and [DUSt3R](https://arxiv.org/abs/2312.14132) if you use this in your research.
- `--save_quantized` to apply 8 bits quantization to the saved checkpoints. It typically reduce ~70% model size with minor quality difference.

### Measuring FPS
```bash
python render.py $OUTPUT_PATH --eval_fps
```

### Rendering views
- Rendering full training views:
    - `python render.py $OUTPUT_PATH --skip_test --rgb_only --use_jpg`
- Rendering testing views and evaluating results:
    - It only works when training with `--eval`.
    - `python render.py $OUTPUT_PATH --skip_train`
    - `python eval.py $OUTPUT_PATH`
- Render fly-through video:
    - `python render_fly_through.py $OUTPUT_PATH`

### Interactive viewer
```bash
python viz.py $OUTPUT_PATH
```
You can then navigate the trained scenes using a web browser. Another interactive viewer is in [example jupyter notebook](./notebooks/example.ipynb) using Kaolin. The FPS of the visualizer is bottleneck by streaming images via network protocal, especially when the it runs on remote server.

https://github.com/user-attachments/assets/44f1737a-c4d3-416d-bdf1-da8b3ecad57b

WebGL is now supported. Thanks [samuelm2](https://github.com/samuelm2) for implementing the [svraster-webgl](https://github.com/samuelm2/svraster-webgl/) viewer.

### Meshing
Remember to train with `--lambda_normal_dmean 0.001 --lambda_normal_dmed 0.001` to get a better geometry. Using sparse depth from COLMAP may also help `--lambda_sparse_depth 0.01`. After the scene optimization completed, run:
```bash
python extract_mesh.py $OUTPUT_PATH
```

### Fusing 2D modality
We can fuse 2D vision foundation feature or sementic segmentation results into voxels easily and instantly. The fusion can naturally smooth out the multi-view inconsistent predictions. More video results are in the [project page](https://svraster.github.io/).
- [demo_segformer.ipynb](./notebooks/demo_segformer.ipynb)
    - We run Segformer to estimate 2D semantic segmentation for all the training views and fuse the 2D semantic into 3D sparse voxel field.
    - ![fusing_segformer](asset/fusing_segformer.jpg)
- [demo_vfm_radio.ipynb](./notebooks/demo_vfm_radio.ipynb)
    - We do Voxel Pooling first as RADIOv2.5 feature is high-dimensional (768 dim). The fusion step is the same as for Segformer.
    - The final fields are a coarser vision foundation model feature field and a density field in original high-resolution.
    - ![fusing_radio](asset/fusing_radio.jpg)

## Experiments on public dataset

**Note:** Be sure to double check the following two experimental details which has non-trivial impact to the quantitative results.
- Ground-truth downsampling: Results from (1) the internal downsampling `--res_downscale` and (2) the preprocessed down-sampled images specified by `--images` are very different. We follow the original 3DGS to use `--images`.
- LPIPS input scale: We follow the original 3DGS to use RGB in range of [0, 1] as default. The correct implementation should be in [-1, 1] which is reported as the corrected LPIPS by `eval.py`.

### Download the 3rd-party processed datasets
- Novel-view synthesis
    - [Mip-NeRF360 dataset](https://jonbarron.info/mipnerf360/)
    - [T&T and DeepBlending dataset](https://github.com/graphdeco-inria/gaussian-splatting#running)
    - [Synthetic NeRF dataset](https://www.matthewtancik.com/nerf/)
    - [Scannet++ dataset](https://kaldir.vc.in.tum.de/scannetpp/)
        - Check [scripts/scannetpp_preproc.py](./scripts/scannetpp_preproc.py) for pre-processing.
- Mesh reconstruction
    - [DTU dataset](https://github.com/Totoro97/NeuS)
        - Check [scripts/dtu_preproc.py](./scripts/dtu_preproc.py) for pre-processing.
    - [Tanks&Temples dataset](https://github.com/hbb1/2d-gaussian-splatting)

### Running base setup
```bash
exp_dir="baseline"
other_cmd_args=""

# Run training
./scripts/mipnerf360_run.sh     output/mipnerf360/baseline     $other_cmd_args
./scripts/synthetic_nerf_run.sh output/synthetic_nerf/baseline $other_cmd_args
./scripts/tandt_db_run.sh       output/tandt_db/baseline       $other_cmd_args
./scripts/dtu_run.sh            output/dtu/baseline            $other_cmd_args
./scripts/tnt_run.sh            output/tnt/baseline            $other_cmd_args

# Summarize results
python scripts/mipnerf360_stat.py     output/mipnerf360/baseline
python scripts/synthetic_nerf_stat.py output/synthetic_nerf/baseline
python scripts/tandt_db_stat.py       output/tandt_db/baseline
python scripts/dtu_stat.py            output/dtu/baseline
python scripts/tnt_stat.py            output/tnt/baseline
```


## Acknowledgement
Our method is developed on the amazing open-source codebase: [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

If you find our work useful in your research, please be so kind to give us a star and citing our paper.
```bibTeX
@article{Sun2024SVR,
  title={Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering},
  author={Cheng Sun and Jaesung Choe and Charles Loop and Wei-Chiu Ma and Yu-Chiang Frank Wang},
  journal={ArXiv},
  year={2024},
  volume={abs/2412.04459},
}
```
