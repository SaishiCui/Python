inFile = open('C:/Users/cuisa/Desktop/python/clustering1.txt', 'r')
data=inFile.readlines()
graph_list=[]
for line in data[1:]:
    u = int(line.split()[0])
    v = int(line.split()[1])
    cost = int(line.split()[2])
    graph_list.append([u,v,cost])
    
    
graph_list = sorted(graph_list, key= lambda x:x[2])


def check_position(node, cluster_list):
    for item in cluster_list:
            if  node in item:
                return cluster_list.index(item) 


    

def clustering_space(graph_list, cluster_list,k):
    recordi=0
    for i in range(len(graph_list)):
        position1=check_position(graph_list[i][0], cluster_list)
        position2=check_position(graph_list[i][1], cluster_list)
    
    
    
        if position1 != position2:
            old_cluster1=cluster_list[position1] 
            old_cluster2=cluster_list[position2]

            new_cluster=old_cluster1+old_cluster2
            cluster_list.remove(old_cluster1)
            cluster_list.remove(old_cluster2)
            cluster_list.append(new_cluster)
    
        recordi +=1
        if len(cluster_list)<k:
            break
    return graph_list[recordi-1][2]


cluster_list=[]
for i in range(1,501):
    number_list = [i]
    cluster_list.append(number_list)

clustering_space(graph_list, cluster_list,4)    














            




# 2. 

vertices = ["".join(x.split(' ')) for x in open('C:/Users/cuisa/Desktop/python/clustering2.txt', 'r').read().split('\n')[1:-1]]



def invert(bit):
    if bit != '0' and bit != '1':
        raise ValueError
    return '1' if bit == '0' else '0'

def similar(v):
    out = []
    for i in range(len(v)):
        out.append(v[:i]+invert(v[i]) + v[i+1:])
        for j in range(i+1, len(v)):
            out.append(v[:i]+invert(v[i])+v[i+1:j]+invert(v[j])+v[j+1:])
    return out


cluster_list=[]
for i in range(1,200001):
    number_list = [i]
    cluster_list.append(number_list)

for i in range(9932,200000):
    for friend in similar(vertices[i]):
        if friend in vertices:
            node=vertices.index(friend)+1
            position1=check_position(node, cluster_list)
            position2=check_position(i+1, cluster_list)
            if position1 != position2:
                old_cluster1=cluster_list[position1] 
                old_cluster2=cluster_list[position2]

                new_cluster=old_cluster1+old_cluster2
                cluster_list.remove(old_cluster1)
                cluster_list.remove(old_cluster2)
                cluster_list.append(new_cluster)
    print(i)
                
                
friend="111001110100111111101101"

for friend in similar(vertices[i]):
    if friend in vertices:
        print(friend)
        
        
        