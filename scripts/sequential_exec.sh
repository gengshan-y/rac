# sequentially run a script over videos
# modified by Gengshan Yang
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#bash scripts/sequential_exec.sh 0 logdir/$seqname-ft3/params_latest.pth \
#        "0 1 2 3 4 5 6 7 8 9 10"'
## argv[1]: gpu id
## argv[2]: sequence name
## argv[3]: weights path
## argv[4]: video id separated by space
## argv[5]: render script path (e.g., scripts/render_vol.sh, scripts/render_nvs.sh)

dev=$1
seqname=$2
modelpath=$3
vid_id=$4
script_path=$5

# Set space as the delimiter
IFS=' '

#Read the split words into an array based on space delimiter
read -a strarr <<< "$vid_id"

for vid in "${strarr[@]}"; do
echo $vid

bash $script_path ${dev} ${seqname} ${modelpath} $vid $vid

done
