#In this Module We Create CustomException file to display the Error Message Customly Created!!
import os,sys
from loggers import logger


def ErrorMessage(message,error_details:sys):
    """
    In Python, the sys module provides detailed information about errors through variables like error_details. 
    From this, we can determine which file the error occurred in and the exact line number.
    However, the error message itself (the description of what went wrong) comes from Python’s Exception class.
    Errors can generally be categorized into two types:
    Human errors (e.g., wrong logic, incorrect input, typos)
    Machine errors (e.g., memory issues, hardware failures, system-level errors).

    """
    #fetching the error_detail coming from sys module
    _,_,error_tb = error_details.exc_info()
    #from error_tb getting filename,lineno of error generate during runtime execution!!
    filename = error_tb.tb_frame.f_code.co_filename
    lineno = error_tb.tb_lineno
    
    #showing custom message of error whenever it occurs
    message = f"Python Error getting in this file {filename} at Line No {lineno} and Message Of Error is {str(message)}"

    return message



#creating customException class
class CustomException(Exception):
    #creating constructor method to initialize the instance variable init
    def __init__(self,message,error_details:sys):

        #calling the parent class constructor and inheriting the Error from it
        super().__init__(message)

        #passing this error message to User defined function
        self.message = ErrorMessage(message=message,error_details=error_details)

        #logging the error here
        logger.error(self.message)

    #creating another Method of CustomException class to display the error Message
    def __str__(self):
        return self.message