import sys
import MySQLdb
from a2audio.roizer import Roizer
from contextlib import closing
import boto.s3.connection
from boto.s3.connection import S3Connection
import json
from a2audio.recanalizer import Recanalizer
from a2audio.samplerates import band2index
import csv
from a2pyutils.logger import Logger
import os
import shutil
from a2pyutils.config import Config
from soundscape.set_visual_scale_lib import *
from classification_lib import create_temp_dir
import time
import multiprocessing
from a2pyutils.jobs_lib import *
import numpy
from joblib import Parallel, delayed
from a2audio.roiset import Roiset
from a2audio.model import Model
import png
import a2pyutils.storage
import json

classificationCanceled =False

def roigen(line,config,tempFolder,jobId,useSsim,bIndex,save_model=True,use_local_storage=False,local_storage_folder=None):
    #print 'roigen'
    global classificationCanceled
    if classificationCanceled:
        return None
    jobId = int(jobId)
    log = Logger(jobId, 'training.py', 'roigen')
    log.also_print = True
    if not save_model:
        log = None
    if use_local_storage:
        storage = a2pyutils.storage.LocalStorage(local_storage_folder)
    else:
        storage = a2pyutils.storage.BotoBucketStorage(config[7], config[4], config[5], config[6])
    db = MySQLdb.connect(host=config[0], user=config[1], passwd=config[2],db=config[3])
    if len(line) < 8:
        db.close()
        log.write("roigen: not enough params")
        return 'err'
    if cancelStatus(db,jobId,tempFolder,False):
        classificationCanceled = True
        quit()
    cancelStatus(db,jobId,tempFolder)
    recId = int(line[0])
    roispeciesId = int(line[1])
    roisongtypeId= int(line[2])
    initTime = float(line[3])
    endingTime = float(line[4])
    lowFreq = float(line[5])
    highFreq = float(line[6])
    recuri = line[7]
    if log:
        log.write("roigen: processing "+recuri)
        log.write("roigen: cutting at "+str(initTime)+" to "+str(endingTime)+ " and filtering from "+str(lowFreq)+" to " + str(highFreq))
    roi = Roizer(recuri,tempFolder, storage,initTime,endingTime,lowFreq,highFreq,log,useSsim,bIndex,save_model)
        
    with closing(db.cursor()) as cursor:
        cursor.execute("""
            UPDATE `jobs` 
            SET `state`="processing", `progress` = `progress` + 1 
            WHERE `job_id` = %s
        """, [
            jobId
        ])
        db.commit()
    if "NoAudio" in roi.status:
        if log:
            log.write("roigen: no audio err " + str(recuri))
        with closing(db.cursor()) as cursor:
            cursor.execute("""
                INSERT INTO `recordings_errors` (`recording_id`, `job_id`) 
                VALUES (%s, %s)
            """, [
                recId, jobId
            ])
            db.commit()
        db.close()
        return 'err'
    else:
        #print 'roigen done'
        if log:
            log.write("roigen: done")
        db.close()
        return [roi,str(roispeciesId)+"_"+str(roisongtypeId)]

def insertRecError(db,jobId,recId):
    with closing(db.cursor()) as cursor:
        cursor.execute("""
            INSERT INTO `recordings_errors` (`recording_id`, `job_id`) 
            VALUES (%s, %s)
        """, [
            recId, jobId
        ])
        db.commit()
    db.close()
    db = None
    
def recnilize(line,config,workingFolder,jobId,pattern,useSsim,useRansac,log=None,bIndex=0,save_model=True,model_type_id=4,use_local_storage=False,local_storage_folder=None):
    global classificationCanceled
    if classificationCanceled:
        return None
    #print 'recnilize'
    if log:
        log.write('analyzing one recording')
    if len(config) < 7:
        #print 'recnilize config error'
        log.write('error analyzing: config is wrong')
        return 'err'
    recId = int(line[5])
    if use_local_storage:
        #print 'using local storage'
        storage = a2pyutils.storage.LocalStorage(local_storage_folder)
    else:
        storage = a2pyutils.storage.BotoBucketStorage(config[7], config[4], config[5], config[6])
    db = None
    conn = None
    try:
        db = MySQLdb.connect(host=config[0], user=config[1], passwd=config[2],db=config[3])
    except:
        if log:
            log.write('error analyzing: Cannot connect to database')
        return 'err'
    with closing(db.cursor()) as cursor:
        cursor.execute("""
            UPDATE `jobs` 
            SET `state`="processing", `progress` = `progress` + 1 
            WHERE `job_id` = %s
        """, [
            jobId
        ])
        db.commit()
    if cancelStatus(db,jobId,workingFolder,False):
        classificationCanceled = True
        quit()
    pid = None
    cancelStatus(db,jobId,workingFolder)
    with closing(db.cursor()) as cursor:
        cursor.execute("""
            SELECT `project_id` 
            FROM `jobs` 
            WHERE `job_id` = %s
        """, [
            jobId
        ])
        db.commit()
        rowpid = cursor.fetchone()
        try:
            pid = rowpid[0]
        except:
            pid = None
    if pid is None:
        insertRecError(db,jobId,recId)
        #print 'recnilize no pid error'
        if log:
            log.write('error analyzing: pid is wrong')
        return 'err'
    key_prefix = 'project_'+str(pid)+'/training_vectors/job_'+str(jobId)+'/'
    recAnalized = None
    #try:
    recAnalized = Recanalizer(line[0] , pattern[0] ,pattern[2] , pattern[3] ,workingFolder, storage,log,False,useSsim,step=16,oldModel =False,numsoffeats=41,ransakit=useRansac,bIndex=bIndex,db=db,rec_id=recId,job_id=jobId,model_type_id=model_type_id)
    #except:
        #log.write('error analyzing: Recanalizer is wrong')
        #insertRecError(db,jobId,recId)
        #return 'err'
    if recAnalized.status == 'Processed':
        recName = line[0].split('/')
        recName = recName[len(recName)-1]
        vectorUri = key_prefix + recName 
        fets = recAnalized.features()
        vector = recAnalized.getVector()
        vectorFile = workingFolder+recName
        if save_model:
            storage.put_file(vectorUri, ','.join(str(x) for x in vector), acl='public-read')
        else:
            fold = local_storage_folder+"/validation-data/job"+str(jobId)
            f = open(fold+'/vectors_'+str(model_type_id)+'_'+str(jobId)+'.csv','a')
            f.write(str(line[3])+','.join(str(x) for x in vector)+"\n")
            f.close()
            
        infos = []
        infos.append(line[4])
        infos.append(line[3])
        infos.append(pattern[4])
        infos.append(pattern[2])
        infos.append(pattern[3])
        infos.append(pattern[1])
        infos.append(line[0])
        infos.append(recId)
        db.close()
        #print 'recnilize done'
        return {"fets":fets,"info":infos}
    else:
        #print 'recnilize done with error'
        if log:
            log.write('error analyzing: recording cannot be analized. status: '+str(recAnalized.status))
        insertRecError(db,jobId,recId)
        if log:
            log.write(line[0])
        return 'err'

def get_training_job_data(db,jobId):
    try:
        with closing(db.cursor()) as cursor:
            cursor.execute("""
                SELECT J.`project_id`, J.`user_id`,
                    JP.model_type_id, JP.training_set_id,
                    JP.validation_set_id, JP.trained_model_id,
                    JP.use_in_training_present,
                    JP.use_in_training_notpresent,
                    JP.use_in_validation_present,
                    JP.use_in_validation_notpresent,
                    JP.name,
                    MT.usesSsim,
                    J.ncpu,
                    MT.usesRansac
                FROM `jobs` J
                JOIN `job_params_training` JP ON JP.job_id = J.job_id , `model_types` MT
                WHERE J.`job_id` = %s and MT.`model_type_id` =  JP.`model_type_id`
            """, [
                jobId
            ])
            row = cursor.fetchone()
    except:
        exit_error("Could not query database with training job #{}".format(jobId),-1,None,jobId,db)
    if not row:
        exit_error("Could not find training job #{}".format(jobId),-1,None,jobId,db)
        
    return  [row['project_id'],
             row['user_id'],
             row['model_type_id'],
             row['training_set_id'],
             row['validation_set_id'],
             row['trained_model_id'],
             row['use_in_training_present'],
             row['use_in_training_notpresent'],
             row['use_in_validation_present'],
             row['use_in_validation_notpresent'],
             row['name'],
             row['usesSsim'],
             row['ncpu'],
             row['usesRansac']
            ]

def get_job_model_type(db,jobId):
    try:
        with closing(db.cursor()) as cursor:
            cursor.execute("""
                SELECT JP.model_type_id
                FROM `jobs` J
                JOIN `job_params_training` JP ON JP.job_id = J.job_id , `model_types` MT
                WHERE J.`job_id` = %s
            """, [
                jobId
            ])
            row = cursor.fetchone()
    except:
        exit_error("Could not query database with training job #{}".format(jobId),-1,None,jobId,db)
    if not row:
        exit_error("Could not find training job #{}".format(jobId),-1,None,jobId,db)
        
    return int(row['model_type_id'])

def get_training_recordings(jobId,training_set_id,workingFolder,log,config,progress_steps):
    db = get_db(config,cursor=False)
    trainingData = []
    """ Training data file creation """
    try:
        with closing(db.cursor()) as cursor:
            # create training file
            cursor.execute("""
                SELECT r.`recording_id`, ts.`species_id`, ts.`songtype_id`,
                    ts.`x1`, ts.`x2`, ts.`y1`, ts.`y2`, r.`uri`
                FROM `training_set_roi_set_data` ts, `recordings` r
                WHERE r.`recording_id` = ts.`recording_id`
                  AND ts.`training_set_id` = %s
            """, [
                training_set_id
            ])
            db.commit()
            trainingFileName = os.path.join(
                workingFolder,
                'training_{}_{}.csv'.format(jobId, training_set_id)
            )
            # write training file to temporary folder
            banwds = []
            with open(trainingFileName, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                numTrainingRows = int(cursor.rowcount)
                progress_steps = numTrainingRows
                for x in range(0, numTrainingRows):
                    rowTraining = cursor.fetchone()
                    banwds.append(float(rowTraining[6])-float(rowTraining[5]))
                    trainingData.append(rowTraining)
                    spamwriter.writerow(rowTraining[0:7+1] + (jobId,))
            meanBand = numpy.mean(banwds)
            maxBand = numpy.max(banwds)
            cursor.execute("""
                SELECT DISTINCT `recording_id`
                FROM `training_set_roi_set_data`
                where `training_set_id` = %s
            """, [
                training_set_id
            ])
            db.commit()
    
            numrecordingsIds = int(cursor.rowcount)
            recordingsIds = []
            for x in range(0, numrecordingsIds):
                rowRec = cursor.fetchone()
                recordingsIds.append(rowRec[0])
    
            cursor.execute("""
                SELECT DISTINCT `species_id`, `songtype_id`
                FROM `training_set_roi_set_data`
                WHERE `training_set_id` = %s
            """, [
                training_set_id
            ])
            db.commit()
    
            numSpeciesSongtype = int(cursor.rowcount)
            speciesSongtype = []
            for x in range(0, numSpeciesSongtype):
                rowSpecies = cursor.fetchone()
                speciesSongtype.append([rowSpecies[0], rowSpecies[1]])
    except:
        exit_error('cannot create training csvs files or access training data from db',-1,log,jobId,db)
        
    cancelStatus(db,jobId,workingFolder)
    
    db.close()
    
    if len(trainingData) == 0 :
        exit_error('cannot create validation csvs files or access validation data from db',-1,log,jobId,db)
    if log:
        log.write('training data gathered')
    
    return trainingData,progress_steps,speciesSongtype,numSpeciesSongtype,maxBand

def get_validation_recordings(workingFolder,jobId,progress_steps,config, storage,log,speciesSongtype,numSpeciesSongtype,project_id,user_id, modelName,
                              useTrainingPresent,useValidationPresent,useTrainingNotPresent,useValidationNotPresent,save_model=True ):
    db = get_db(config,cursor=False)
    validationData = []
    validationId = None
    """ Validation file creation """
    try:
        validationFile = workingFolder+'/validation_'+str(jobId)+'.csv'
        with open(validationFile, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for x in range(0, numSpeciesSongtype):
                spst = speciesSongtype[x]
                with closing(db.cursor()) as cursor:
                    cursor.execute("""
                        (SELECT r.`uri` , `species_id` , `songtype_id` , `present` , r.`recording_id`
                        FROM `recording_validations` rv, `recordings` r
                        WHERE r.`recording_id` = rv.`recording_id`
                          AND rv.`project_id` = %s
                          AND `species_id` = %s
                          AND `songtype_id` = %s
                          AND `present` = 1
                          ORDER BY rand()
                          LIMIT %s)
                          UNION
                        (SELECT r.`uri` , `species_id` , `songtype_id` , `present` , r.`recording_id`
                        FROM `recording_validations` rv, `recordings` r
                        WHERE r.`recording_id` = rv.`recording_id`
                          AND rv.`project_id` = %s
                          AND `species_id` = %s
                          AND `songtype_id` = %s
                          AND `present` = 0
                          ORDER BY rand()
                          LIMIT %s)
                    """, [
                        project_id, spst[0], spst[1], (int(useTrainingPresent)+int(useValidationPresent )) ,
                        project_id, spst[0], spst[1], (int(useTrainingNotPresent)+int(useValidationNotPresent )) 
                    ])
                    
                    db.commit()
    
                    numValidationRows = int(cursor.rowcount)
    
                    progress_steps = progress_steps + numValidationRows
    
                    for x in range(0, numValidationRows):
                        rowValidation = cursor.fetchone()
                        cc = (str(rowValidation[1])+"_"+str(rowValidation[2]))
                        validationData.append([rowValidation[0] ,rowValidation[1] ,rowValidation[2] ,rowValidation[3] , cc ,rowValidation[4]])
                        spamwriter.writerow([rowValidation[0] ,rowValidation[1] ,rowValidation[2] ,rowValidation[3] , cc ,rowValidation[4]])
        
        # compute storage key
        valiKey = 'project_{}/validations/job_{}.csv'.format(project_id, jobId)    
        # save validation file to storage
        if save_model:
            storage.put_file_path(valiKey, validationFile)
    
        # save validation to DB
        progress_steps = progress_steps + 15
        with closing(db.cursor()) as cursor:
            cursor.execute("""
                INSERT INTO `validation_set`(
                    `validation_set_id`, `project_id`, `user_id`, `name`, `uri`,
                    `params`, `job_id`
                ) VALUES (
                    NULL, %s, %s, %s, %s, %s, %s
                )
            """, [
                project_id, user_id, modelName+" validation", valiKey,
                json.dumps({'name': modelName}),
                jobId
            ])
            db.commit()
            validationId = cursor.lastrowid
            cursor.execute("""
                UPDATE `job_params_training`
                SET `validation_set_id` = %s
                WHERE `job_id` = %s
            """, [
                cursor.lastrowid, jobId
            ])
            db.commit()
            
            cursor.execute("""
                UPDATE `jobs`
                SET `progress_steps` = %s, progress=0, state="processing"
                WHERE `job_id` = %s
            """, [
                progress_steps, jobId
            ])
            db.commit()
    except:
        exit_error('cannot create validation csvs files or access validation data from db',-1,log,jobId,db)
        
    cancelStatus(db,jobId,workingFolder)
    
    if len(validationData) == 0 :
        exit_error('cannot create validation csvs files or access validation data from db',-1,log,jobId,db)
    
    db.close()
    if log:
        log.write('validation data gathered')
    
    return validationData,validationId
   
def generate_rois(trainingData,num_cores,config,workingFolder,jobId,useSsim,bIndex,log,db,save_model=True,use_local_storage=False,local_storage_folder=None):
    rois = None
    """Roigenerator"""
    try:
        rois = Parallel(n_jobs=2)(delayed(roigen)(line,config,workingFolder,jobId,useSsim,bIndex,save_model,use_local_storage,local_storage_folder) for line in trainingData)
    except:
        exit_error('roigenerator failed',-1,log,jobId=jobId,db=db,workingFolder=workingFolder)
        
    if rois is None or len(rois) == 0 :
        exit_error('cannot create rois from recordings',-1,log,jobId,db,workingFolder)
    
    errRoi = 0
    for roi in rois:
        if 'err' in roi:
            errRoi = errRoi +1
            
    if errRoi == len(rois):
        exit_error('cannot create rois from recordings (all error)',-1,log,jobId,db,workingFolder)
        
    cancelStatus(db,jobId,workingFolder)
    if log:
        log.write('rois generated')
    
    return rois

from pylab import *
def rois_2_surface(rois,log,bIndex,useSsim,db,jobId,workingFolder,number_of_rois_to_align=None):
    patternSurfaces = {}
    classes = {}
    """Align rois"""
    if log:
        log.write('rois 2 surface')
    try:
        for roi in rois:
            if 'err' not in roi:
                classid = roi[1]
                lowFreq = roi[0].lowF
                highFreq = roi[0].highF
                sample_rate = roi[0].sample_rate
                spec = roi[0].spec
                rows = spec.shape[0]
                columns = spec.shape[1]
                if classid in classes:
                    classes[classid].addRoi(float(lowFreq),float(highFreq),float(sample_rate),spec,rows,columns)
                else:
                    classes[classid] = Roiset(classid,float(sample_rate) ,log, (not useSsim))
                    classes[classid].addRoi(float(lowFreq),float(highFreq),float(sample_rate),spec,rows,columns)
        for i in classes:
            classes[i].alignSamples(bIndex,number_of_rois_to_align)
            patternSurfaces[i] = [classes[i].getSurface(),classes[i].setSampleRate,classes[i].lowestFreq ,classes[i].highestFreq,classes[i].maxColumns]

    except:
         exit_error('cannot align rois',-1,log,jobId,db,workingFolder)
            
    cancelStatus(db,jobId,workingFolder)
    
    if len(patternSurfaces) == 0 :
        exit_error('cannot create pattern surface from rois',-1,log,jobId,db,workingFolder)
    if len(classes) == 0 :
        exit_error('classes were not defined',-1,log,jobId,db,workingFolder)
        
    return classes,patternSurfaces

def analyze_recordings(validationData,log,num_cores,config,workingFolder,jobId,patternSurfaces,useSsim,useRansac,bIndex,db,save_model,model_type_id,use_local_storage=False,local_storage_folder=None):
    results = None
    """Recnilize"""
    if log:
        log.write("analizing recordings")
    
    #try:
    results = Parallel(n_jobs=2)(delayed(recnilize)(line,config,workingFolder,jobId,(patternSurfaces[line[4]]),useSsim,useRansac,log,bIndex,save_model,model_type_id,use_local_storage,local_storage_folder) for line in validationData)
    #except:
        #exit_error('cannot terminate parallel loop (analyzing recordings)',-1,log,jobId,db,workingFolder)

    if results is None:
        exit_error('cannot analyze recordings',-1,log,jobId,db,workingFolder)
    if log:
        log.write("recs analized")
    
    cancelStatus(db,jobId,workingFolder)
    
    presentsCount = 0
    ausenceCount = 0
    for res in results:
        if 'err' not in res:
            if int(res['info'][1]) == 0:
                ausenceCount = ausenceCount + 1
            if int(res['info'][1]) == 1:
                presentsCount = presentsCount + 1            
            
    if presentsCount < 2 or ausenceCount < 2:
        exit_error('not enough validations to create model',-1,log,jobId,db,workingFolder)
        
    return results,presentsCount,ausenceCount

def add_samples_to_model(results,jobId,db,workingFolder,log,patternSurfaces,model_type=4):
    models = {}
    if log:
        log.write('adding samples to model')
    try:
        for res in results:
            if 'err' not in res:
                classid = res['info'][0]
                if classid in models:
                    models[classid].addSample(res['info'][1],res['fets'],res['info'][6],res['info'][7])
                else:
                    models[classid] = Model(classid,patternSurfaces[classid][0],jobId,model_type)
                    models[classid].addSample(res['info'][1],res['fets'],res['info'][6],res['info'][7])
    except:
        exit_error('cannot add samples to model',-1,log,jobId,db,workingFolder)
    
    cancelStatus(db,jobId,workingFolder)
    if log:
        log.write('model has samples')
    return models

def balance_validation_samples(useTrainingPresent,useValidationPresent,useTrainingNotPresent,useValidationNotPresent, presentsCount,ausenceCount):
    if (useTrainingPresent+useValidationPresent) > presentsCount:
        if presentsCount <= useTrainingPresent:
            useTrainingPresent = presentsCount - 1
            useValidationPresent = 1
        else:
            useValidationPresent = presentsCount - useTrainingPresent

    if (useTrainingNotPresent + useValidationNotPresent)  > ausenceCount:
        if ausenceCount <= useTrainingNotPresent:
            useTrainingNotPresent = ausenceCount - 1
            useValidationNotPresent = 1
        else:
            useValidationNotPresent = ausenceCount  - useTrainingNotPresent
    return   useTrainingPresent,useValidationPresent,useTrainingNotPresent,useValidationNotPresent

def train_model(model,useTrainingPresent,useTrainingNotPresent,useValidationPresent,useValidationNotPresent,log,jobId,db,workingFolder,useSsim,useRansac,bIndex,patternSurfaces,classId):
    print "train models"
    if log:
        log.write("training model")
    modelFilesLocation = workingFolder
    resultSplit = False
    try:
        resultSplit = model.splitData(useTrainingPresent,useTrainingNotPresent,useValidationPresent,useValidationNotPresent)
    except:
        exit_error('error spliting data for validation',-1,log,jobId,db,workingFolder)
    if not resultSplit:
        return None
    print "model train"
    try:
        model.train()
    except:
        exit_error('error training model',-1,log,jobId,db,workingFolder)

    validationsLocalFile = modelFilesLocation+'job_'+str(jobId)+'_vals.csv'    
    if useValidationPresent > 0:
        try:
            model.validate()
            model.saveValidations(validationsLocalFile)
        except:
           exit_error('error validating model',-1,log,jobId,db,workingFolder)
    
    print "model save"   
    modFile = modelFilesLocation+"model_"+str(jobId)+"_"+str(classId)+".mod"
    if log:
        log.write("saving model to file")
    try:
        model.save(modFile,patternSurfaces[2] ,patternSurfaces[3],patternSurfaces[4],useSsim,useRansac,bIndex)
    except:
        exit_error('error saving model file to local storage',-1,log,jobId,db,workingFolder)
        
    modelStats = None
    dimsPat = None
    try:
        modelStats = model.modelStats()
        dimsPat = model.getPatternDim()
        specpat = model.getSpec()
    except :
        exit_error('cannot get stats from model',-1,log,jobId,db,workingFolder)       
    spid = classId.split('_')[0]
    if log:
        log.write("k fold validation")
    validation_k_fold = True
    foldesn = 10
    print "k fold"
    if validation_k_fold:
        data = specpat
        specToShow = numpy.zeros(shape=(0,int(data.shape[1])))
        rowsInSpec = data.shape[0]
        spec = numpy.copy(data)
        if  numpy.sum(numpy.sum(spec == -10000))>0:
            spec[spec == -10000] = numpy.nan
        for j in range(0,rowsInSpec):
            if abs(numpy.nansum(spec[j,:])) > 0.0:
                specToShow = numpy.vstack((specToShow,numpy.copy(spec[j,:])))
        
        totalData,totalPos ,totalNeg ,accuracy_score,precision_score,sensitivity_score,specificity_score = model.k_fold_validation(folds=foldesn,db=db,jobId=jobId,pshape=specToShow.shape,speciesId=spid)
        with closing(db.cursor()) as cursor:
            cursor.execute("""INSERT INTO `k_fold_Validations`(`job_id`, `totaln`, `pos_n`, `neg_n`, `k_folds`, `accuracy`, `precision`, `sensitivity`, `specificity`,`w`,`h`)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [jobId,totalData,totalPos ,totalNeg ,foldesn,accuracy_score,precision_score,sensitivity_score,specificity_score,dimsPat[1],dimsPat[0] ])
            db.commit()
    if log:
        log.write("done")

    return modelStats
from pylab import *
def prepare_png_data(data,log,jobId,db,workingFolder):
    if log:
        log.write('preparing png data')

    try:
        specToShow = numpy.zeros(shape=(0,int(data.shape[1])))
        rowsInSpec = data.shape[0]
        spec = numpy.copy(data)
        if  numpy.sum(numpy.sum(spec == -10000))>0:
            spec[spec == -10000] = numpy.nan
        for j in range(0,rowsInSpec):
            if abs(numpy.nansum(spec[j,:])) > 0.0:
                specToShow = numpy.vstack((specToShow,numpy.copy(spec[j,:])))
        if numpy.sum(numpy.sum(numpy.isnan(specToShow)))>0:
            specToShow[numpy.isnan(specToShow)] = numpy.nanmean(numpy.nanmean(specToShow))
        smin = numpy.min(numpy.min(specToShow))#numpy.min([numpy.min((specToShow[j])) for j in range(specToShow.shape[0])])
        smax = numpy.max(numpy.max(specToShow))# numpy.max([numpy.max((specToShow[j])) for j in range(specToShow.shape[0])])
        matrix = 255*(1-((specToShow - smin)/(smax-smin)))
    except:
        exit_error('cannot prepare png data',-1,log,jobId,db,workingFolder)
    if log:
        log.write('png data prepared')
    return matrix 

def save_model_to_db(classId,db,jobId,training_set_id,modelStats,patternSurfaces,pngKey,modelname,model_type_id,modKey,project_id,user_id,valiId,log,workingFolder):
    species,songtype = classId.split("_")
    try:
        #save model to DB
        with closing(db.cursor()) as cursor:
            cursor.execute("""
                UPDATE `jobs` 
                SET `state`="processing", `progress` = `progress` + 5 
                WHERE `job_id` = %s
            """, [
                jobId
            ])
            db.commit()
            cursor.execute("""
                SELECT   max(ts.`x2` -  ts.`x1`) as l , min(ts.`y1`)  as min, max(ts.`y2`) as max
                FROM `training_set_roi_set_data` ts
                WHERE ts.`training_set_id` =  %s
            """, [
                training_set_id
            ])
            db.commit()
            row = cursor.fetchone()
            lengthRoi = row['l']	
            minFrequ = row['min']
            maxFrequ = row['max']
            
            cursor.execute("""
                SELECT count(*) as c 
                FROM `training_set_roi_set_data` ts 
                WHERE ts.`training_set_id` =  %s
            """, [
                training_set_id
            ])
            db.commit()
            row = cursor.fetchone()
            totalRois = row['c']
            
            
            statsJson = json.dumps({
                "roicount":totalRois, "roilength": lengthRoi, "roilowfreq": minFrequ, "roihighfreq":maxFrequ,
                "accuracy":modelStats[0], "precision":modelStats[1], "sensitivity": modelStats[2],
                "forestoobscore": modelStats[3], "roisamplerate": patternSurfaces[1], "roipng": pngKey,
                "specificity": modelStats[5], "tp": modelStats[6], "fp": modelStats[7],
                "tn": modelStats[8], "fn": modelStats[9], "minv": modelStats[10], "maxv": modelStats[11]
            })
        
            cursor.execute("""
                INSERT INTO `models`(`name`, `model_type_id`, `uri`, `date_created`, `project_id`, `user_id`, `training_set_id`, `validation_set_id`)
                VALUES (%s, %s, %s, now(), %s, %s, %s, %s)
            """, [
                modelname, model_type_id, modKey, project_id, user_id, training_set_id, valiId
            ])
            db.commit()
            insertmodelId = cursor.lastrowid
            
            cursor.execute("""
                INSERT INTO `model_stats`(`model_id`, `json_stats`) 
                VALUES (%s, %s)
            """, [
                insertmodelId, statsJson
            ])
            db.commit()
            
            cursor.execute("""
                INSERT INTO `model_classes`(`model_id`, `species_id`, `songtype_id`) 
                VALUES (%s, %s, %s)
            """, [
                insertmodelId, species, songtype
            ])
            db.commit()       
            
            cursor.execute("""
                UPDATE `job_params_training` 
                SET `trained_model_id` = %s 
                WHERE `job_id` = %s
            """, [
                insertmodelId, jobId
            ])
            db.commit()
            
            cursor.execute("""
                UPDATE `jobs` 
                SET `last_update` = now() 
                WHERE `job_id` = %s
            """, [
                jobId
            ])
            db.commit()
            cursor.execute("""
                UPDATE `jobs` 
                SET `state`="completed", `progress` = `progress_steps` ,  `completed` = 1 , `last_update` = now() 
                WHERE `job_id` = %s
            """, [
                jobId
            ])
            db.commit()
    except:
        exit_error('error saving model into database',-1,log,jobId,db,workingFolder)
        

def train_pattern_matching(db,jobId,log,config, storage,save_model=True,model_type_id=4,use_local_storage=False,local_storage_folder=None,number_of_rois_to_align=None):
    (
        project_id, user_id,
        model_type_id, training_set_id,
        validation_set_id, trained_model_id,
        use_in_training_present,
        use_in_training_notpresent,
        use_in_validation_present,
        use_in_validation_notpresent,
        name,
        ssim_flag,
        ncpu,
        ransac_flag
    ) = get_training_job_data(db,jobId)
    
    num_cores = multiprocessing.cpu_count()
    if int(ncpu) > 0:
        num_cores = int(ncpu)
    
    if use_local_storage:
        fold = local_storage_folder+"/validation-data/job"+str(jobId)
        if not os.path.exists(fold):
            try:
                os.makedirs(fold)
            except:
                pass
            
    progress_steps = 0
    print "create temp dir"
    workingFolder = create_temp_dir(jobId,log)
    
    cancelStatus(db,jobId,workingFolder)
    print "get training recordings"
    training_recordings,progress_steps,speciesSongtype,numSpeciesSongtype,maxBand = get_training_recordings(jobId,training_set_id,workingFolder,log,config,progress_steps)
    
    cancelStatus(db,jobId,workingFolder)
    print "get validation recordings"
    validation_recordings,validationId = get_validation_recordings(workingFolder,jobId,progress_steps,config, storage, log,speciesSongtype,numSpeciesSongtype,project_id,user_id,name,use_in_training_present,use_in_validation_present,use_in_training_notpresent,use_in_validation_notpresent,save_model)
    
    cancelStatus(db,jobId,workingFolder)
    print "band 2 index"
    bIndex = band2index(maxBand)
    
    cancelStatus(db,jobId,workingFolder)
    print "generate rois"
    rois =  generate_rois(training_recordings,num_cores,config,workingFolder,jobId,ssim_flag,bIndex,log,db,save_model,use_local_storage,local_storage_folder)
    
    cancelStatus(db,jobId,workingFolder)
    print "rois to surface"
    classes,patternSurfaces = rois_2_surface(rois,log,bIndex,ssim_flag,db,jobId,workingFolder,number_of_rois_to_align)
    
    cancelStatus(db,jobId,workingFolder)
    print "analyze recordings"
    recordings_results,presentsCount,ausenceCount = analyze_recordings(validation_recordings ,log,num_cores,config,workingFolder,jobId,patternSurfaces,ssim_flag,ransac_flag,bIndex,db,save_model,model_type_id,use_local_storage,local_storage_folder)
    
    cancelStatus(db,jobId,workingFolder)
    print  "add samples to model"
    models = add_samples_to_model(recordings_results,jobId,db,workingFolder,log,patternSurfaces,model_type_id)
    
    cancelStatus(db,jobId,workingFolder)
    print "balance validation"
    use_in_training_present,use_in_validation_present,use_in_training_notpresent,use_in_validation_notpresent = balance_validation_samples(use_in_training_present,use_in_validation_present,use_in_training_notpresent,use_in_validation_notpresent, presentsCount,ausenceCount)
    
    cancelStatus(db,jobId,workingFolder)    
    print "for models"
    modelSaved = False
    for classId in models:
        modelStats = train_model(models[classId],use_in_training_present,use_in_training_notpresent,use_in_validation_present,use_in_validation_notpresent,log,jobId,db,workingFolder,ssim_flag,ransac_flag,bIndex,patternSurfaces[classId],classId)
        
        if save_model:
            pngFilename = workingFolder+'job_'+str(jobId)+'_'+str(classId)+'.png'
           
            patternPngMatrix = prepare_png_data(modelStats[4],log,jobId,db,workingFolder)
            
            png.from_array(patternPngMatrix, 'L;8').save(pngFilename)
            
            pngKey = 'project_'+str(project_id)+'/models/job_'+str(jobId)+'_'+str(classId)+'.png'
            modKey = 'project_'+str(project_id)+'/models/job_'+str(jobId)+'_'+str(classId)+'.mod'
            files2upload = {
                'model':{'key':modKey ,
                         'file':workingFolder+"model_"+str(jobId)+"_"+str(classId)+".mod",
                         'public':False},
                'validation':{'key':'project_'+str(project_id)+'/validations/job_'+str(jobId)+'_vals.csv',
                              'file':workingFolder+'job_'+str(jobId)+'_vals.csv',
                              'public':False},
                'png':{'key': pngKey ,
                       'file':pngFilename,
                       'public':True}
            }
            upload_files_2storage(storage, files2upload,log,jobId,db,workingFolder)
        
            save_model_to_db(classId,db,jobId,training_set_id,modelStats,patternSurfaces[classId],pngKey,name,model_type_id,modKey,project_id,user_id,validationId,log,workingFolder)
        else:
            fold = local_storage_folder+"/pattern-data/job"+str(jobId)
            if not os.path.exists(fold):
                os.makedirs(fold)
            pngFilename = fold+'/'+'job_'+str(jobId)+'_'+str(classId)+'.png'
    
            patternPngMatrix = prepare_png_data(modelStats[4],log,jobId,db,workingFolder)
            
            png.from_array(patternPngMatrix, 'L;8').save(pngFilename)
        if log:  
            log.write("model saved")
        modelSaved = True
    
    if os.path.exists(workingFolder):
        shutil.rmtree(workingFolder)
    return modelSaved 

def run_training(jobId,save_model=True,use_local_storage=False,local_storage_folder=None,number_of_rois_to_align=None):
    log=None
    try:
        configuration = Config()
        config = configuration.data()
        db = get_db(config)
        retValue = False
        start_time = time.time()   
        log = Logger(jobId, 'training.py', 'main')
        log.also_print = True
        log.write('training job id'+str(jobId))
        log.write('fetching config.')
        log.write('config fetched.')
        log.write('fectching storage.')
        if use_local_storage:
            storage = a2pyutils.storage.LocalStorage(local_storage_folder)
        else:
            storage = a2pyutils.storage.BotoBucketStorage(config[7], config[4], config[5], config[6])
        log.write('storage fetched.')
        log.write('training database connection succesful')
        model_type_id = get_job_model_type(db,jobId)
        log.write('job model type fetched.')
    except:
        exit_error('There was an error initializing job', code=-1, log=log,jobId=jobId,db=db,workingFolder=None,doExit=False)
        return False
    if model_type_id in [1,2,3,4,5,6]:
        log.write("Pattern Matching (modified Alvarez thesis)")
        #if not save_model:
            #log = None
        retValue = train_pattern_matching(db,jobId,log,config, storage,save_model,model_type_id,use_local_storage,local_storage_folder,number_of_rois_to_align)
        if not retValue:
            exit_error('There was an error running pattern matching job', code=-1, log=log,jobId=jobId,db=db,workingFolder=None,doExit=False)
        db.close()
        return retValue
    else:
        log.write("incorrect model type")
        exit_error('Error: incorrect model type', code=-1, log=log,jobId=jobId,db=db,workingFolder=None,doExit=False)
        db.close()
        return False
