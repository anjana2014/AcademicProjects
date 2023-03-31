import os          #import os for python functions
import xmlrpc.client  #for the remote procedure call for client
from src.client.client import proxy #proxy for the web request
from src.client import config as clientConfig  #confi for the client

print('hi, welcome to distributed systems') #printing a welcome message

if __name__ == "__main__":
    print('app running')  #checking whether app is running (or) not

    proxy.File_DELETE('jerry.jpg') #deleting the file bob.jpg
    print('File Deletion successful') #priting the message for deleting file 

    with open(os.path.join(clientConfig.client_directory_path, 'Kick.jpg'), 'rb') as file: #giving the file path for uploading for the client
        proxy.File_UPLOAD('Kick.jpg', xmlrpc.client.Binary(file.read())) #uploading the file mickey.jpg to the client
        print('Upload Successful') # file uploaded to the server automatically as changes are made in client

    with open(os.path.join(clientConfig.client_directory_path, 'Kick.jpg'), 'wb') as file:# opening the file
        file.write(proxy.File_DOWNLOAD('Kick.jpg')) #writing the data in to the server
        print('Download Successful') # printing the downladed message as it is updated in the server

    proxy.File_RENAME('snoopy.jpg', 'snoopy6.jpg') #renaming the file
    print('Rename Successful') #printing success message



