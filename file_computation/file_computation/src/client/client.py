import os #import os for the python modules
import time #import time
import xmlrpc.client #import client 
from src.client import config #import config from the client

proxy = xmlrpc.client.ServerProxy(config.serverURL)  # proxy request from client to server for the communication

def checkFilePresent(fileName): # checking whether the file is present in the client (or) not
    return os.path.isfile(os.path.join(config.client_directory_path, fileName)) #returing the file path 

def fileUpload(fileName): # uploading the files function
    try:
        with open(os.path.join(config.client_directory_path, fileName), 'rb') as file:#file path for getting the files
            proxy.File_UPLOAD(fileName, xmlrpc.client.Binary(file.read()))#performing the upload function
            print(f"after uploading file {fileName}") # printing uploaded function name
    except:
        print(f"uploading failed for {fileName}") #printing if there is any exception is created (or) not
    return

def fetchFiles(): # getting the files  from the directory
    files = [file for file in os.listdir(config.client_directory_path) if (checkFilePresent(file) and file != '__init__.py')]# getting all the files
    return files#returing the files

def uploadModifiedFiles(files): # uploading the modified files
    for file in files: # loop for checking the files
        fileUpload(file) # uploading the files 
    return

def updateFileChangeTime(files, modificationTime):# getting the files after modifying
   
    for file in files: # loop for checking the files
        modificationTime[file] = os.path.getmtime(os.path.join(config.client_directory_path, file)) # path after modifying the files
    return

def fetchModifiedFiles(files, modificationTime, lastCheckTime):  #getting the modified files
    mod_files= []
    for file in files:#loop for checking the modified files
        if(modificationTime[file] >= lastCheckTime): # last updated files
             mod_files.append(file)#appending the files
    return  mod_files # returing the modiied files

def fetchDeletedFiles(files, modificationTime):#deleting the files
    deletedFiles = []
    for file in modificationTime: # checking the for the modified files
        if file not in files:
            deletedFiles.append(file)#getting the files after deleting the files
    return deletedFiles# returning the deleting the files


def deleteFiles(files, modificationTime): # deleting the files in the server also
    for file in files: #checking the all the files
       try:
        proxy.File_DELETE(file) #calling the delete function in client
        del modificationTime[file]  #deleting the files
       except:
        print(f"deleting failed for {file}") #printing the if there any exception
    return

def fileCheck(lastCheckTime, modificationTime):  # checking the for the last updated and modified files

    files = fetchFiles() # checking the files
    print(files)

    updateFileChangeTime(files, modificationTime) #updating the file modified times
    

    mod_files = fetchModifiedFiles(files, modificationTime, lastCheckTime) #modified files
    
    uploadModifiedFiles(mod_files) #uploading the modified files

    deletedFiles = fetchDeletedFiles(files, modificationTime) # getting the deleted files 
   
    deleteFiles(deletedFiles, modificationTime)  # deleting the files in server    
    return

def client_start(): #starting the client_file_inspector when we are running the server to copy files from client to server.

    files = fetchFiles() #getting all the files
    uploadModifiedFiles(files) # uploading the modified files
    modificationTime = {} # getting the file modified times
    lastCheckTime = time.time()  #checking the last checked timings
    while True: 
        print('started the file inspector') # stating the inspector
        startTime = time.time() #checking the start time
        print(lastCheckTime, startTime) #printing the last_checked and checked time
        fileCheck(lastCheckTime, modificationTime)  #checking if there any modifications in the last_checked and checked time
        lastCheckTime = startTime  # making the lastCheckTime as startTime time 
        print(lastCheckTime, startTime) #printing the lastCheckTime and start_time
        del startTime #deleting the  startTime time
        time.sleep(config.refresh_interval)
    return    
