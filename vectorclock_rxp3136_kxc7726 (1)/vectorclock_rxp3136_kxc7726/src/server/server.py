import os   #import operatig systems for pyth module
from socketserver import ThreadingMixIn  #import threading 
from xmlrpc.server import SimpleXMLRPCServer #import xmlrmpc server funcxtion 
from src.client import client #import client 
import random  #import random 
class MultiThreadedSimpleXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass

vectclock = []#function for vector clock
port_single = 0 #single port
log_clk = 0 #function for clock
ports_for_process = [] #multiple ports

def increm_vectclock():  #function to increment the vector clock 
    global vectclock     #declaration of vectclock var 
    portindx = get_func_portindex() #port index function to get the port index 
    vectclock[portindx] += 1 #vector clock incrementation using the port index 
    return

def get_func_portindex(): #function to get the port index 
    global port_single, ports_for_process  #variables declaration for single and multiple ports
    for i, first_port in enumerate(ports_for_process):  #for first port in all these processes
        if first_port == port_single: #if first port is equal to the port given 
            return i   #return 
    raise Exception("port for process not found in the set of ports") #excepytion if the port not found
    return

def update_vectclk(vector):  #update the vector clock function after message
    global vectclock       #vector clock declaration
    if (len(vector) != len(vectclock)): #if the length of the vector is not equal to the length of the vector clock 
        raise Exception("vector size mismatch")   #raisedddd excetion for vector mismatch
    vectclock = [max(v1,v2) for v1,v2 in zip(vector, vectclock)]  #if the vector clock is  in zip of vector and vector clock 
    return

def initialize_synchronization(): #initialize synchronization
    global ports_for_process #declaration of ports
    proxies = [client.get_proxy(x) for x in ports_for_process] #proxies for the ports
    logical_clocks = [proxy.get_logclk() for proxy in proxies] #get the proxy function using clock 
    average_clock = int(sum(logical_clocks)/len(logical_clocks)) #using the func with the formual
    [proxy.synch_clock(average_clock, len(ports_for_process)) for proxy in proxies]#proxy clock for synch using clk
    return

def get_logclk():  #get function for the clock 
    global log_clk #glb declaration of the  variable
    return log_clk #return function for the clock

def incom_msg(vector):  #incomming msg function 
    global port_single  #declaring the variable port
    print(f"process node on port {port_single} received a message")  #process node function for the message received 
    print_vectclk()#print vectclk func call
    increm_vectclock()#increment vectclk func call
    update_vectclk(vector) #update vectclk function call using the vector
    print_vectclk() #calling the printvector clock function
    return

def print_logical_clock(): #print function
    #print(f"logical clock for {port_single}: {log_clk}")
    return #return val

def synch_clock(counter, n): #synchclock for this one 
    global vectclock #vect clk declaration
    global log_clk  #declaration of function
    log_clk = counter #calling the counter
    vectclock = [counter for x in range(n)] #vectclock range
    print_logical_clock() #print the function call
    return

def initialize_logclock(): #initialize funtion
    global log_clk #declaring the variable here 
    log_clk = random.randint(1, 10)  #initializing some integer for this function
    return

def outgng_msg(target_port): #outgoing msg function declaration using the target port
    global port_single, vectclock #declaration of vars using port and vectclk
    print_vectclk()  #calling the print func
    increm_vectclock() #incrementing the vect clk func call
    proxy = client.get_proxy(target_port) ##get port func
    proxy.incom_msg(vectclock) #get proxy for the incomming msg func
    print(f"process node on port {port_single} sent a message")  #prcess node sent a msg function
    print_vectclk()#pprint vect func call
    return

def outgng_broadcastmsg(target_port): #outgoing msg function declaration using the target port
    global port_single, vectclock #declaration of vars using port and vectclk
    print_vectclk()  #calling the print func
    increm_vectclock() #incrementing the vect clk func call
    for p in range(len(target_port)):
        proxy = client.get_proxy(target_port[p]) ##get port func
        proxy.incom_msg(vectclock) #get proxy for the incomming msg func
    print(f"process node on port {port_single} sent a  message")  #prcess node sent a msg function
    print_vectclk()#pprint vect func call
    
    return

def print_vectclk(): #print vector clock function 
    print(f"vector clock for {port_single}: {vectclock}") #for the single port print the function
    return

def func_to_start_server(data_for_servers): #func to start server function declaration 
    global port_single, ports_for_process # declaration of teh global variables
    port_single, ports_for_process = data_for_servers #single port and  multiple ports
    initialize_logclock() #initialize the clock
    print_logical_clock()   #print func for the clock
    with MultiThreadedSimpleXMLRPCServer(('localhost', port_single), allow_none= True) as server: #with the thraeding server
        server.register_introspection_functions() #server functions
        server.register_function(synch_clock) #server reg functions 
        server.register_function(get_logclk) #ser reg function for the lgclk
        server.register_function(initialize_synchronization)#initialze synch for gthe clk
        server.register_function(incom_msg) #for the incomming msg and 
        server.register_function(outgng_msg) #fr the outgoing msg
        server.register_function(outgng_broadcastmsg) #fr the outgoing msg
        print(f"server is serving on port {port_single}") #server print
        server.serve_forever() #server func
    return



    
