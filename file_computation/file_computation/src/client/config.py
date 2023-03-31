import os
import src.client.data as content
from src.server import config as serverConfig

serverURL = f"http://localhost:{serverConfig.portNumber}" #server URL
refresh_interval = 8 #the files are refreshed every 8 seconds
client_directory_path = os.path.dirname(os.path.abspath(content.__file__) ) #directorty name for the client
print(f"The client is taking care of the files in this directory: {client_directory_path}") #the directory handled by the client