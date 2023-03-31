import sys
import time
import os
import copy
import numpy as np
from node_class import Node

#read the command line arguments
start_filename = sys.argv[1]
goal_filename = sys.argv[2]
method = sys.argv[3]
dump_flag = sys.argv[4]
trace_file = open('trace.txt', 'w')

class SearchAlgorithm:
    def __init__(self, start_node, goal_node, dump_flag):
        self.start_node = start_node
        self.goal_node = goal_node
        self.dump_flag = dump_flag
        self.nodes_expanded = 0
        self.nodes_popped = 0
        self.nodes_generated = 0
        self.max_fringe_size = 0
        self.fringe = []
        self.closed_set = []
        if self.dump_flag:
            self.trace_file = trace_file
            self.trace_file.write('Command-Line Arguments : ' + str(sys.argv) + '\n')
            self.trace_file.write('Method Selected: ' + method + '\n')

    def run(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_solution(self, node):
        solution = []
        while node.parent:
            solution.append(node)
            node = node.parent
        solution.append(node)
        return solution[::-1]

    def print_stats(self, node):
        print('Nodes Expanded: ' + str(self.nodes_expanded))
        print('Nodes Popped: ' + str(self.nodes_popped))
        print('Nodes Generated: ' + str(self.nodes_generated))
        print('Max Fringe Size: ' + str(self.max_fringe_size))
        print('Solution Found at depth ' + str(node.depth) + ' with cost ' + str(node.path_cost))
        print('Steps:')
        for step in self.get_solution(node)[1:]:
            print('   ', step.action)


class bfs(SearchAlgorithm):
    def run(self):
        self.fringe.append(self.start_node)
        while self.fringe:
            node = self.fringe.pop(0)
            self.nodes_popped += 1
            if node == self.goal_node:
                self.print_stats(node)
                return
            if self.dump_flag == 'true':
                self.trace_file.write('Generating successors to ' + str(node) + '\n')
            if node not in self.closed_set:
                self.closed_set.append(node)
                children = node.expand(self.goal_node)
                self.fringe.extend(children)
                self.nodes_expanded += 1
                self.nodes_generated += len(children)
                if len(self.fringe) > self.max_fringe_size:
                    self.max_fringe_size = len(self.fringe)
                    self.sort_fringe()
                if self.dump_flag == 'true':
                    self.trace_file.write('    ' + str(len(children)) + ' successors generated'+ '\n')
                    self.trace_file.write('    Closed Set: ' + str(self.closed_set)+ '\n')
                    self.trace_file.write('    Fringe: ' + str(self.fringe)+ '\n')
        print('No solution found')
    
    def sort_fringe(self):
        pass

class a_star(bfs):
    def sort_fringe(self):
        self.fringe.sort(key=lambda x: x.f_value, reverse=False)

class ucs(bfs):
    def sort_fringe(self):
        self.fringe.sort(key=lambda x: x.path_cost, reverse=False)

class greedy(bfs):
    def sort_fringe(self):
        self.fringe.sort(key=lambda x: x.h_value, reverse=False)
        
class dfs(SearchAlgorithm):

    def __init__(self, start_node, goal_node, dump_flag, max_depth):
        super().__init__(start_node, goal_node, dump_flag)
        self.max_depth = max_depth
        self.done = 0

    def run(self):
        self.fringe.append(self.start_node)
        while self.fringe:
            node = self.fringe.pop(0)
            self.nodes_popped += 1
            if node.depth > self.max_depth:
                break
            if node == self.goal_node:
                self.print_stats(node)
                self.done = 1
                return
            if self.dump_flag == 'true':
                self.trace_file.write('Generating successors to ' + str(node) + '\n')
            if node not in self.closed_set:
                self.closed_set.append(node)
                children = node.expand(self.goal_node)
                self.fringe = children + self.fringe
                self.nodes_expanded += 1
                self.nodes_generated += len(children)
                if len(self.fringe) > self.max_fringe_size:
                    self.max_fringe_size = len(self.fringe)
                if self.dump_flag == 'true':
                    self.trace_file.write('    ' + str(len(children)) + ' successors generated'+ '\n')
                    self.trace_file.write('    Closed Set: ' + str(self.closed_set)+ '\n')
                    self.trace_file.write('    Fringe: ' + str(self.fringe)+ '\n')
        print('No solution found')


class ids(dfs):
    def run(self):
        max_depth = self.max_depth
        for depth in range(1, self.max_depth):
            print('Depth: ' + str(depth))
            self.max_depth = depth
            dfs.run(self)
            if self.done == 1:
                return


def main():
    #read the start file and goal file
    start_file = open(start_filename, 'r')
    goal_file = open(goal_filename, 'r')

    # make start node and goal node
    start_node = Node(start_file)
    goal_node = Node(goal_file)

    #call the appropriate search method
    if method == 'bfs':
        search = bfs(start_node, goal_node, dump_flag)
        search.run()
    elif method == 'ucs':
        search = ucs(start_node, goal_node, dump_flag)
        search.run()
    elif method == 'a_star':
        search = a_star(start_node, goal_node, dump_flag)
        search.run()
    elif method == 'greedy':
        search = greedy(start_node, goal_node, dump_flag)
        search.run()
    elif method == 'dfs':
        search = dfs(start_node, goal_node, dump_flag, 100000)
        search.run()
    elif method == 'dls':
        limit = int(sys.argv[5])
        search = dfs(start_node, goal_node, dump_flag, limit)
        search.run()
    elif method == 'ids':
        search = ids(start_node, goal_node, dump_flag, 100000)
        search.run()
    else:
        search = a_star(start_node, goal_node, dump_flag)
        search.run()

main()