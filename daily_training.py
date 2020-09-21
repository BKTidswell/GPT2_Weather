import gpt_2_simple as gpt2
from datetime import datetime
from random import *
import shutil, os, glob
from google.cloud import logging

logging_client = logging.Client()
# The name of the log to write to
log_name = 'my-log'
# Selects the log to write to
logger = logging_client.logger(log_name)

 
def moveAllFilesinDir(srcDir, dstDir):
	# Check if both the are directories

	if os.path.isdir(srcDir) and os.path.isdir(dstDir):
		# Iterate over all the files in source directory
		for filePath in glob.glob(srcDir + '/*'):
			# Move each file to destination Directory
			shutil.move(filePath, dstDir)
		text = "Finished moving "+srcDir+" to "+ dstDir
	else:
		text = "srcDir & dstDir should be Directories"

	print(text)
	logger.log_text(text)

def trainNewModel():
			
	#Step 1: Check if it has been 24 hours

	logger.log_text("Started Training") 

	textfiles = ["Emily_D_poems.txt","SeaChute_Bisuness.txt","Magnus_Statements.txt"]

	#Step 2: Select the text file
	file_name = "text_files/"+textfiles[randint(0,len(textfiles)-1)]
	RUN_NAME = "run1"
	MODEL_NAME= "run1"


	#Step 3: Move and date the old run1 model into a backup folder

	# #Step 3a: set up backup folder
	cwd = os.getcwd()
	now = datetime.now()
	# date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
	# path = cwd+"/model_backups/"+date_time
	# os.mkdir(path)

	# #Step 3b: Move old model
	# moveAllFilesinDir(cwd+"/models/run1",path)

	#Real Step 3: Delete old model

	if cwd == "/":
		cwd = ""

	shutil.rmtree(cwd+"/models/run1") 
	os.mkdir(cwd+"/models/run1")

	#Step 4: Move the run1 checkpoint into the models
	moveAllFilesinDir(cwd+"/checkpoint/run1/",cwd+"/models/run1")

	#Step 4a: Delete the old run1 folder
	os.rmdir(cwd+"/checkpoint/run1")

	logger.log_text("Moved Files")

	files = os.listdir("/")
	logger.log_text(str(files)) 

	files2 = os.listdir(cwd+"models/run1")
	files3 = os.listdir(cwd+"checkpoint")

	logger.log_text(str(files2)) 
	logger.log_text(str(files3)) 

	#Step 5: Train the model
	sess = gpt2.start_tf_sess()

	try:
		gpt2.finetune(sess,
		              dataset=file_name,
		              model_name= MODEL_NAME,
		              steps=10,
		              restore_from='fresh',
		              run_name=RUN_NAME,
		              print_every=10,
		              sample_every=10,
		              save_every=10,
		              learning_rate = 1e-5,
		              overwrite=False,
		              )
	except:
		logger.log_text("Got here I guess") 

	#Step 6: Log this training sessions
	text = "%s : Trained with %s \n" % (now.strftime("%m/%d/%Y %H:%M:%S"), file_name)
	logger.log_text(text) 
	print('Logged: {}'.format(text))
	# f = open("training_log.txt", "a")
	# f.write("%s : Trained with %s \n" % (now.strftime("%m/%d/%Y %H:%M:%S"), file_name))
	# f.close()
