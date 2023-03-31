import os
import xmlrpc.client
from socketserver import ThreadingMixIn
from xmlrpc.server import SimpleXMLRPCServer
from src.server import config

class ServerClass(ThreadingMixIn, SimpleXMLRPCServer):
    pass

def server_start(): # using multithreading in server
    with ServerClass(('localhost', config.portNumber), allow_none= True) as serverObj:
        print(f"server runs on port {config.portNumber}") # checking if the port is working
        serverObj.register_introspection_functions()
        #registering the functions to this object
        serverObj.register_function(File_UPLOAD) 
        serverObj.register_function(File_DOWNLOAD)
        serverObj.register_function(File_RENAME) 
        serverObj.register_function(File_DELETE)
        serverObj.serve_forever() #serves until the connection is intact
    return

def File_UPLOAD(fileName: str, fileContent) -> None: #file upload
    with open(os.path.join(config.server_directory_path, fileName), 'wb') as doc:
        doc.write(fileContent.data)  #writing the file to the directory in that path
    return

def File_DOWNLOAD(fileName: str): # file download
    with open(os.path.join(config.server_directory_path, fileName), 'rb') as doc:
        return xmlrpc.client.Binary(doc.read()) #reads the file

def File_DELETE(fileName: str) -> None: #delete file
    newpath = os.path.join(config.server_directory_path, fileName) #path where file will be removed
    os.remove(newpath) #using remove funtion to delete file
    print(f"deletion successful:{newpath}")
    return
    
def File_RENAME(presentName: str, newName: str) -> None: #file rename
    presentFilepath = os.path.join(config.server_directory_path, presentName) # path in which the file to be renamed is taken from
    newFilepath = os.path.join(config.server_directory_path, newName)  #the new path where the renamed file will be placed
    os.rename(presentFilepath, newFilepath) #path renaming
    return





    
