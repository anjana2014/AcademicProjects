import multiprocessing as mp     #multiprocessig for more than 1 processor
import threading                  #threads import
import random                    #import random
import time                      #import time 
from src.client import client    #import client server
from src.server.server import func_to_start_server #server impotrt function 

ports_for_process = [3900,3901,3902]   #ports to be used

def initialize_nodes():      #initialize_nodes function 
    global ports_for_process   #declaring the ports variable
    print("*****starting nodes******")  #starting the nodes
    with mp.Pool(processes=len(ports_for_process)) as pool: #length of ports as pool
            pool.map(func_to_start_server, [(x,ports_for_process) for x in ports_for_process]) #mapping the ports
    return

def echo(data_for_servers):     #data_for_servers function 
    port_single, ports_for_process = data_for_servers   #for single and multiple ports 
    print(port_single, ports_for_process) #print the ports 
    return port_single       #returning the port

def trigger_message():       #trigger msg function 
    global ports_for_process  #declaration for [ports] 
    first_port = random.choice(ports_for_process) #first port
    second_port = get_different_port(first_port)   #second port
    proxy1 = client.get_proxy(first_port)     #declaring the proxy for client
    print("****** triggering a new message event *******")
    print(f"process node on port {first_port} is sending message to node on port {second_port}")  #sending message to the second port
    proxy1.outgng_msg(second_port) #second port 
    return

def broadcast_message():       #trigger msg function 
    global ports_for_process  #declaration for [ports] 
    first_port = random.choice(ports_for_process) #first port
    second_port =[]
    second_port=get_all_different_port(first_port)  #second port
    proxy1 = client.get_proxy(first_port) #declaring the proxy for client  
    print("****** triggering a BroadCast message event *******")
    print(f"process node on port {first_port} is sending message to different nodes on ports {second_port}")  #sending message to the second port
    proxy1.outgng_broadcastmsg(second_port) #second port 
    return

def get_different_port(first_port): #get port function 
    global ports_for_process    #variable for ports 
    if len(ports_for_process) < 2:  #if length is less for ports 
        return
    second_port = random.choice(ports_for_process)   #second port for process 
    if second_port!= first_port:  #if second port is not equal to first port
        return second_port       #return second port 
    return get_different_port(first_port) #get different port

def get_all_different_port(first_port): #get port function 
    global ports_for_process    #variable for ports 
    if len(ports_for_process) < 2:  #if length is less for ports 
        return
    rem_port=[]
    for p in range(len(ports_for_process)):  #second port for process 
        if ports_for_process[p]!= first_port:  #if second port is not equal to first port
           rem_port.append(ports_for_process[p])  
            
    return rem_port#return second port
            
     #get different port

if __name__ == "__main__":#main function
    try:   #try 
        t1 = threading.Thread(target=initialize_nodes)  #declaring the t1 var
        t1.start()  #starting thraed 1  
        time.sleep(1) #sleep for thraed 1   
        randport= random.choice(ports_for_process) #if the port is same 
        randproxy = client.get_proxy(randport) #proxy for rand port
        randproxy.initialize_synchronization() #initialize synch
        input=input("Enter 1 for Unicast Message or 2 for  BroadCast Message:")
        if (int(input) ==1):
            trigger_message() #trigger message 
        if(int(input)==2):
            broadcast_message()
        
    except:
        print('exiting')#exit the app

    