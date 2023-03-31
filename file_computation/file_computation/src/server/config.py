import os 
import src.server.data as content

portNumber = 8023 #server port number
server_directory_path = os.path.dirname(os.path.abspath(content.__file__)) #path to the directory in server
print(f"The server is taking care of the files in this directory:  {server_directory_path}") #the directory handled by the server