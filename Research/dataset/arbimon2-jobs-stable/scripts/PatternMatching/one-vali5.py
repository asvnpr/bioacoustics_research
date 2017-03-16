import sys

sys.path.append("/home/rafa/node/arbimon2-jobs-stable/lib/")

local_storage_folder="/home/rafa/recs/"

from a2audio.training_lib import run_training
from soundscape.set_visual_scale_lib import get_db
from a2pyutils.config import Config
import json
from contextlib import closing

numOfRois = 5

j = sys.argv[1]
retVal = run_training(int(j),False,use_local_storage=True,local_storage_folder=local_storage_folder,number_of_rois_to_align=numOfRois)
