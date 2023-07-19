# a script to call mesh extarction and mesh rendering
#bash scripts/extract_and_render_mesh.sh 0 dog80 logdir/dog80-v1 0
dev=$1
seqname=$2
model_path=$3
vidid=$4   # pose traj
rootid=$vidid  # root traj

testdir=${model_path%/*} # %: from end
sample_grid3d=256
add_args="--sample_grid3d ${sample_grid3d} --mc_threshold 0 \
  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

prefix=$testdir/$seqname-{$vidid}

# extrat meshes
CUDA_VISIBLE_DEVICES=$dev python scripts/extract_mesh.py --flagfile=$testdir/opts.log \
                  --seqname $seqname \
                  --model_path $model_path \
                  --test_frames {$vidid} \
                  --nolineload \
                  $add_args
                  #--noce_color \


# reference view
suffix=
trgpath=$prefix$suffix
rootdir=$trgpath-ctrajs-
CUDA_VISIBLE_DEVICES=$dev python scripts/render_mesh.py --testdir $testdir \
                     --outpath $trgpath --vp 0 --render_bg \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid}
CUDA_VISIBLE_DEVICES=$dev python scripts/render_mesh.py --testdir $testdir \
                     --outpath $trgpath-bne --vp 0 --render_bg \
                     --gray_color --vis_bones \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid}

python scripts/render_mesh.py --testdir $testdir \
                     --outpath $prefix-vid \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid} \
                     --append_img yes \
                     --append_render no
