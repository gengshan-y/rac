# helper function to parallelize render jobs over gpus
# python scripts/parallel.py "0" cat76 cat76 scripts/extract_and_render_mesh.sh
import sys
import configparser
import pdb
import os

devs=sys.argv[1]
seqname=sys.argv[2]
loadname=sys.argv[3]
scriptpath=sys.argv[4]

devs=devs.split(",")
num_dev = len(devs)

config = configparser.RawConfigParser()
config.read('configs/%s.config'%seqname)

model_path = 'logdir/%s/params_latest.pth'%loadname

vid_groups = {}
for vidid in range(len(config.sections())-1):
    dev = devs[vidid%num_dev]
    if dev in vid_groups.keys():
        vid_groups[dev] += ' %s'%vidid
    else:
        vid_groups[dev] = '%s'%vidid

for dev in devs:
    cmd = 'bash scripts/sequential_exec.sh %s %s %s \'%s\' %s '%(dev, seqname, model_path, vid_groups[dev], scriptpath)
    cmd = 'screen -dmS "render-%s-%s" bash -c "%s"'%(seqname, dev, cmd)
    print(cmd)
    err = os.system(cmd)
    if err:
        print("FATAL: command failed")
        sys.exit(err)
