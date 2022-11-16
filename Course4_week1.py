
import math
import numpy as np


def readgraph(path):
    inFile = open(path, 'r')
    data=inFile.readlines()
    graph_dic = dict()
    n=int(data[0].split()[0])
    for line in data[1:]:
        start_node = int(line.split()[0]) 
        end_node = int(line.split()[1])
        length = float(line.split()[2])
        
        if start_node not in graph_dic: 
            graph_dic[ start_node ] = [[],[]]
            graph_dic[start_node][0].append(end_node)
            graph_dic[start_node][1].append(length)
        else:
            graph_dic[start_node][0].append(end_node)
            graph_dic[start_node][1].append(length)
    return graph_dic, n



def floyed_warshall(graph_dic, n):        
    A_k0 = np.array([[float(0)]*n]*n)
    
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i == j:
                A_k0[i-1][j-1]= float(0)
            elif j in graph_dic[i][0]:
                idx =graph_dic[i][0].index(j)
                A_k0[i-1][j-1]=graph_dic[i][1][idx]
            elif j not in graph_dic[i][0]:
                A_k0[i-1][j-1]=math.inf
                
    A_k_old = np.array(A_k0)
    A_k_new =  np.array([[float(0)]*n]*n)
    detect= 0
    for k in range(1,n):
        for i in range(n):
            for j in range(n):
                A_k_new[i,j] = min(A_k_old[i,j], A_k_old[i,k]+A_k_old[k,j] )
        
        for i in range(n):
            if A_k_new[i][i]<0:
                print("Detected a negative cycle \n")
                print("negative cycle is ", [i,A_k_new[i][i]])
                detect = 1
                break
        if detect == 1:
            break
        A_k_old =  np.array(A_k_new)
        print(k)
        
    if detect ==1:
        return print("Negative cycle")
    else:
        min_path = min(min(row) for row in A_k_new)
        return min_path
        
graph_dic1 = readgraph('C:/Users/cuisa/Desktop/python/APSP1.txt')[0]
n = readgraph('C:/Users/cuisa/Desktop/python/APSP1.txt')[1]

graph_dic2 = readgraph('C:/Users/cuisa/Desktop/python/APSP2.txt')[0]
n = readgraph('C:/Users/cuisa/Desktop/python/APSP2.txt')[1]

graph_dic3 = readgraph('C:/Users/cuisa/Desktop/python/APSP3.txt')[0]
n = readgraph('C:/Users/cuisa/Desktop/python/APSP3.txt')[1]


floyed_warshall(graph_dic1, n)
floyed_warshall(graph_dic2, n)
floyed_warshall(graph_dic3, n)



