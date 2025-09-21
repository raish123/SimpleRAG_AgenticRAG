import logging
import os,sys
from pathlib import Path
from datetime import datetime


#creating a log directories to store the logs file init!!!
log_dir = 'Logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir,exist_ok=True)


#creating a logs-file with timestamp
timestamp = datetime.now().strftime(format="%Y_%m_%d_%H_%M_%S")

log_filename = f"logs_{timestamp}.log"

#creating a filepath for this logs file
log_filepath = os.path.join(log_dir,log_filename)


FORMAT = "[%(asctime)s]-%(levelname)s-%(lineno)d-%(message)s"

#configuring the logging module
logging.basicConfig(level= logging.INFO,
    format = FORMAT,
    handlers=[
        logging.FileHandler(log_filepath,mode="a",encoding="utf-8"),
        logging.StreamHandler(sys.stdout) #this line show/print log on terminal
    ]
    
)

#creating a logging object
logger = logging.getLogger(name = "RAG_Logging")