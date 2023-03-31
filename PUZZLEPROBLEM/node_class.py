import copy
import numpy as np

class Node:
    def __init__(self, file):
        self.parent = None
        self.path_cost = 0
        self.depth = 0
        self.state = []
        if file is not None:
            #read the file
            for i in range(3):
                line = file.readline()
                line = line.split()
                self.state.append(line)
            file.close()
            #convert the state to integers
            self.state = np.array(self.state, dtype=int)
        self.heuristic_value = 0
        self.f_value = 0
        self.action = None
    
    def new_child(self, row, col, i, j, action, goal_node=None):
        #create a new node
        child = Node(None)
        child.state = copy.deepcopy(self.state)
        child.state[row][col], child.state[row + i][col + j] = child.state[row + i][col + j], child.state[row][col]
        child.parent = self
        child.path_cost = self.path_cost + child.state[row][col]
        child.depth = self.depth + 1
        child.action = 'Move ' + str(child.state[row][col]) + ' '  + action
        if goal_node is not None:
            child.heuristic_value = np.sum(np.abs(child.state - goal_node.state))
        child.f_value = child.path_cost + child.heuristic_value
        return child

    def expand(self, goal_node=None):
        #initialize the children
        children = []
        #find the blank tile
        blank_tile = np.where(self.state == 0)
        #get the row and column of the blank tile
        row = blank_tile[0][0]
        col = blank_tile[1][0]
        #check if the blank tile can be moved up or down or left or right
        if row > 0:
            children.append(self.new_child(row, col, -1, 0, 'Down', goal_node))
        if row < 2:
            children.append(self.new_child(row, col, 1, 0, 'Up', goal_node))
        if col > 0:
            children.append(self.new_child(row, col, 0, -1, 'Right', goal_node))
        if col < 2:
            children.append(self.new_child(row, col, 0, 1, 'Left', goal_node))
        return children
    
    def __eq__(self, other):
        return np.array_equal(self.state, other.state)
    
    def __str__(self):
        rep = '< state = ' + str(self.state.tolist()) 
        rep += ', action = {' + str(self.action) + '}' 
        rep += ' g(n) = ' + str(self.path_cost) 
        rep += ', d = ' + str(self.depth) 
        rep += ', f(n) = ' + str(self.f_value) 
        rep += ', Parent = Pointer to ' + str(self.parent) + ' >'
        return rep
    
    def __repr__(self):
        return self.__str__()
     