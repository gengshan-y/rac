# RAC: Reconstructing Animatable Categories from Videos

**[[Project page]](https://gengshan-y.github.io/rac-www/)**

![rac](https://user-images.githubusercontent.com/13134872/236699509-ee1569ba-28da-4e53-aedc-ff20cc19e87f.gif)


### Updates
- [05/07/2023] We plan to release an re-implementation of training code through [lab4d](https://github.com/lab4d-org/lab4d).
- [01/22/2023] This repo is under development. It will contain the pre-trained category models of cats, dogs, and human.


### Install
```
git clone git@github.com:gengshan-y/rac.git --recursive
cd rac

# base dependencies
conda env create -f misc/rac-cu113.yml

# other dependencies
conda activate rac-cu113
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

### Re-targeting
```
python explore.py --flagfile logdir/dog80-v0/opts.log --nolineload --seqname dog80 --full_mesh --noce_color --svid 69 --tvid 45
```
retargets the source video 69 to target video 45. Results are saved at `logdir/dog80-v0/explore-motion-69.mp4`.

https://user-images.githubusercontent.com/13134872/236701581-4b34131a-aee7-45b3-9b3e-63b4af4940e5.mp4



### Demo
See `demo.ipynb` for an interactive demo visualizing learned morphology and articulations.
![Screenshot 2023-01-22 at 9 58 34 PM](https://user-images.githubusercontent.com/13134872/213958804-a78f2a17-bea6-46ac-8a9c-8e321ff4df44.png)
