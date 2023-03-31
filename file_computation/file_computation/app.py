import threading #import threding for the threads
from src.server.server import server_start #starting the server file
from src.client.client import client_start #starting teh client file

if __name__ == "__main__":
    try:
        serverThread = threading.Thread(target=server_start)#creating a thread for server
        serverThread.start()#starting the server thread
        print('server has begun')#checking server thread 
        clientThread = threading.Thread(target=client_start) #creating a thread for client
        clientThread.start()#checking client thread
        print('file inspection has begun') 
    except:
        print('execution is terminated') #execution is terminated

    