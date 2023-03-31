import os   #import operating systems 
import time  #import time 
import xmlrpc.client  #import client

def get_proxy(port_single):  #get proxy for the port
    server_uri = f"http://localhost:{port_single}"  #server uri for the port
    proxy = xmlrpc.client.ServerProxy(server_uri)  #proxy for the server uri
    return proxy #return proxy function
