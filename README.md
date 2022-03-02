# Scene_Graph_Parsing
Pretraining scene graph parsing tasks for [symbolic goal learning](https://github.com/ivalab/mmf) of robotic manipulation.

## Installation
1. Create a conda environment
```
conda create --name sgp python=3.6
```
2. Activate built conda environment
```
conda activate sgp
```
3. Compile everything by running ```make``` in the main directory. 

## Usage
Before starting training anything, you will need to 
1. Update the config.py file which contains paths to the dataset.
2. Add the current path to your PYTHONPATH: ```export PYTHONPATH=/path_to_folder/Scene_Graph_Parsing```
3. Download the [dataset]() and follow the instruction to unzip downloaded file to the ```data``` folder. 

The first step is to train the object detector:
```
python models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/object_detector -nepoch 50 -ngpu 1 -nwork 3 -p 100 -clip 5 -resnet -val_size 3223
```
The second step is to train the scene graph classification:
```
python models/train_rels_gt.py -m sgcls -model rtnet -order leftright -nl_obj 2 -nl_edge 4 -b 3 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/saved_folder/gt-xx.tar -save_dir checkpoints/sgcls -nepoch 50 -val_size 3222 -resnet -limit_vision
```
To be noticed, the training of scene graph detection is not performed in this work since the aim of pretraining on scene graph parsing is to 
help the network interpret relationships between object. Generating proper Region-of-interest is not our goal.

Lastly, to evaluate trained model for scene graph classification, run
```
python models/eval_rels.py -m sgcls -model rtnet -order leftright -nl_obj 2 -nl_edge 4 -b 3 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -resnet -limit_vision -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/sgcls/vgrel-xx.tar
```

## License
GKNet is released under the MIT License (refer to the LICENSE file for details). Portions of codes are borrowed from [Neural motifs](https://github.com/rowanz/neural-motifs). Please refer to the regional License of this project.

## Citation
If you find this work is helpful and use it in your work, please cite:

```bibtex
@article{xu2022sgl,
  title={SGL: Symbolic Goal Learning for Human Instruction Following in Robot Manipulation},
  author={Xu, Ruinian and Chen, Hongyi and Lin, Yunzhi, and Vela, Patricio A},
  journal={arXiv preprint arXiv:2202.12912},
  year={2022}
}
```
