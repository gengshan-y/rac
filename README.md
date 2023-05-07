# RAC: Reconstructing Animatable Categories from Videos

**[[Project page]](https://gengshan-y.github.io/rac-www/)**

![rac](https://user-images.githubusercontent.com/13134872/236699509-ee1569ba-28da-4e53-aedc-ff20cc19e87f.gif)


### Updates
- [05/07/2023] We plan to release re-implementation of training code through [lab4d](https://github.com/lab4d-org/lab4d).
- [01/22/2023] This repo is under development. It will contain the pre-trained category models of cats, dogs, and human.


### Install

We recommend using mamba to install, which is much faster conda in resolving conflicts.
To install mamba, do 
```conda install -c conda-forge mamba -y```
Then you may replace `conda install` with `mamba install`

```
git clone git@github.com:gengshan-y/rac.git --recursive
cd rac

# base dependencies
mamba env create -f misc/rac-env.yml -y

# other dependencies
conda activate rac
pip install git+https://github.com/pytorch/functorch.git@a6e0e61
pip install git+https://github.com/facebookresearch/pytorch3d.git
cd quaternion; python setup.py install; cd -
```

### Pretrained models
```
# download model weights
wget https://www.dropbox.com/sh/h1w82lb4rg48jui/AACD8q-DCFjyDhRx0-j7EjWLa -O tmp.zip
mkdir -p logdir
unzip tmp.zip -d ./logdir
rm tmp.zip
```

### Shape interplation
```
python explore.py --flagfile logdir/dog80-v0/opts.log --nolineload --seqname dog80 --full_mesh --noce_color --svid 69 --tvid 45 --interp_beta
```
It interpolates the shape between the source video 69 and target video 45. Results are saved at `logdir/dog80-v0/explore-interp-69.mp4`.

![interp](https://user-images.githubusercontent.com/13134872/236706537-89627d39-e044-4312-8142-d16eb6b87c40.gif)

### Re-targeting
```
python explore.py --flagfile logdir/dog80-v0/opts.log --nolineload --seqname dog80 --full_mesh --noce_color --svid 69 --tvid 45
```
It retargets the source video 69 to target video 45. Results are saved at `logdir/dog80-v0/explore-motion-69.mp4`.

![retarget](https://user-images.githubusercontent.com/13134872/236706546-bbb50529-ea0e-4726-8414-c466738b304a.gif)

### Demo
See `demo.ipynb` for an interactive demo visualizing learned morphology and articulations.
![Screenshot 2023-01-22 at 9 58 34 PM](https://user-images.githubusercontent.com/13134872/213958804-a78f2a17-bea6-46ac-8a9c-8e321ff4df44.png)
