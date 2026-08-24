# synthetic-to-real-pipeline-for-agricultural-robotics
## Repository structure

* `assets/` — Blender models and sugar beet 3D assets
* `synthetic_generation/` — Isaac Sim assets and dataset generation scripts
* `dataset/` — real and synthetic datasets used in the experiments
* `perception/` — RT-DETR training, configuration and export scripts
* `evaluation/` — dataset and model evaluation utilities
* `tools/` — supporting utilities, mainly for USD inspection and material fixes

## Pipeline

The project follows the workflow:

3D asset preparation → Isaac Sim synthetic data generation → dataset preparation → RT-DETR training → evaluation → deployment
