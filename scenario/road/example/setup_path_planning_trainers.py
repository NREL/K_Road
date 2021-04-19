# Idea here is this is a minimal script that can live on /scratch and drive a run.
# This will be used as a template for fiddling with parameters below and submitting separate jobs for each set

import glob
import json

import numpy as np
from cavs_environments.vehicle.deep_road.deep_road_constants import DeepRoadConstants

from scenario import do_train


def run_one(clb, db, cpb, sb, mnpl, mnsd, mxsd, mxsa):
    inp = {}
    rewarder_config = {'collision_bonus': clb, 'distance_bonus': db, 'completion_bonus': cpb, 'step_bonus': sb}
    process_config = {"min_plan_length": mnpl, "min_step_dist": mnsd, "max_step_dist": mxsd, "max_step_angle": mxsa}
    inp['name'] = 'path_planning'
    inp['scenario'] = 'avoid_bots'
    inp['rewarder_config'] = rewarder_config
    inp['process_config'] = process_config
    # inp dict provides overrides for the default config
    # so off we go...
    do_train(inp)


batchstr = """#!/bin/bash

#SBATCH --account=cavs
#SBATCH --time=4:00:00
#SBATCH --job-name=plannniing
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#####SBATCH --partition=debug

 
module purge
module load conda
 
source ~/.bashrc
 
conda activate ray
 
module purge
which python
 
unset LD_PRELOAD
export CARLA_HOME=/projects/cavs/Carla/carla-0.9.10/
export TUNE_RESULT_DIR=/scratch/pgraf/ray_results/
 
python run_train_path_planning.py --inpfile=%s 
"""


def setup_one(clb, db, cpb, sb, mnpl, mnsd, mxsd, mxsa):
    inp = {}
    rewarder_config = {'collision_bonus': clb, 'distance_bonus': db, 'completion_bonus': cpb, 'step_bonus': sb}
    process_config = {"min_plan_length": mnpl, "min_step_dist": mnsd, "max_step_dist": mxsd, "max_step_angle": mxsa}
    inp['name'] = 'path_planning'
    inp['scenario'] = 'avoid_bots'
    inp['rewarder_config'] = rewarder_config
    inp['process_config'] = process_config
    # inp dict provides overrides for the default config
    # so off we go...
    pstr = "%d.%d.%d.%d.%d" % (clb, db, cpb, sb, mnpl)
    inpname = "inp.%s" % pstr
    batchname = "batch.%s.sh" % pstr
    batch = batchstr % inpname
    fout = open(inpname, "w")
    json.dump(inp, fout)
    fout.close()
    fout = open(batchname, "w")
    fout.write(batch)
    fout.close()


def submit_batch(folder):
    files = glob.glob("batch*.sh")
    for f in files:
        print("queuing", f)
        os.system("sbatch %s" % f)


def setup_runs():
    # for range of vals of range of params:
    #   - write a run script that has those params in it
    #   - write a batch script that calls the run script
    clbs = [-1, -10, -100]
    dbs = [1, 5]
    cpbs = [50]
    sbs = [1]
    mnpls = [10, 40]
    mnsd = DeepRoadConstants.car_length
    mxsd = DeepRoadConstants.car_length * 3
    mxsa = 35 * np.pi / 180.0
    for clb in clbs:
        for db in dbs:
            for cpb in cpbs:
                for sb in sbs:
                    for mnpl in mnpls:
                        setup_one(clb, db, cpb, sb, mnpl, mnsd, mxsd, mxsa)


if __name__ == "__main__":
    setup_runs()
