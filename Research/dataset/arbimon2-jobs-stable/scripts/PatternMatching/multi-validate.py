#! .env/bin/python

import sys

sys.path.append("/home/rafa/node/arbimon2-jobs-stable/lib/")

local_storage_folder="/home/rafa/recs/"

from a2audio.training_lib import run_training
from soundscape.set_visual_scale_lib import get_db
from a2pyutils.config import Config
import json
from contextlib import closing


model_Types = [1,2,3,4]

if len(sys.argv) > 1:
    model_Types=[]
    model_Types.append(int(sys.argv[1]))
    
numOfRois = 5
if len(sys.argv) > 2:
    numOfRois = (int(sys.argv[2]))
    
configuration = Config()
config = configuration.data()
db = get_db(config)

validation_data = None

with open('scripts/data/validation_data.json') as fd:
    validation_data = json.load(fd)

rows = []
allidss = []
for model_type in model_Types:
    job_ids = []
    for i in range(len(validation_data)):
        r = validation_data[i]
        with closing(db.cursor()) as cursor:
            cursor.execute("""INSERT INTO `jobs`
                            ( `job_type_id`, `date_created`, `last_update`, `project_id`,
                            `user_id`, `uri`)
                            VALUES
                            (1,now(),now(),33,
                            1,'')
                            """)
            db.commit()
            jobId = cursor.lastrowid
            allidss.append(jobId)
            job_ids.append(jobId)
            cursor.execute("""INSERT INTO `job_params_training`
                (`job_id`, `model_type_id`, `training_set_id`, `use_in_training_present`,
                `use_in_training_notpresent`, `use_in_validation_present`, `use_in_validation_notpresent`, `name`)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                [jobId, str( model_type),
                 r['t_set'] , r['present'] , r['not_present'],
                 r['absent'],r['not_absent'] , r['name']
                 ])
            db.commit()

print allidss
quit()

    # for i in range(len(validation_data)):
        # r = validation_data[i]
        # j = job_ids[i]
        # retVal = run_training(int(j),False,use_local_storage=True,local_storage_folder=local_storage_folder,number_of_rois_to_align=numOfRois)
        # # row = None
        # # row1= None
        # # if retVal:
        # #     with closing(db.cursor()) as cursor:    
        # #         cursor.execute("SELECT  `totaln`, `pos_n`, `neg_n`, `k_folds`, `accuracy`, `precision`, `sensitivity`, `specificity` ,`w`,`h` FROM `k_fold_Validations` WHERE `job_id` = "+str(j))
        # #         row = cursor.fetchone()
        # #         db.commit()
        # #     with closing(db.cursor()) as cursor:    
        # #         cursor.execute("select avg(exec_time) as time from recanalizer_stats where job_id = "+str(j)+"")
        # #         row1 = cursor.fetchone()
        # #         db.commit()        
        # #     rows.append(','.join([str(model_type),r['name'],str(row['totaln']),str(row['pos_n']),str(row['neg_n']),str(row['k_folds']),str(row['accuracy']),str(row['precision']),str(row['sensitivity']),str(row['specificity']),str(row1['time']),str(row['w']),str(row['h']) , str(int(row['h'])*int(row['w']))]))
        # #     row = None
        # #     row1 = None
        # # else:
        # #     print 'job failed'
        # # f = open(local_storage_folder+'/results_'+str(j)+"_"+str(model_type)+".csv",'w')
        # # f.write(','.join(['model_type','species','total_n','pos','neg','k','accuracy','precision','sensitivity','specificity','exec_time','width','height','total_pixels'])+"\n")
        # # for r in rows:
        # #     f.write(r+'\n')
        # # f.close() 

# print ','.join(['model_type','species','total_n','pos','neg','k','accuracy','precision','sensitivity','specificity','exec_time','width','height','total_pixels'])
# for r in rows:
    # print r
