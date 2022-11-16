## order by difference ##
from collections import Counter

weight = []
length = []
with open("C:/Users/cuisa/Desktop/python/scheduling_jobs.txt") as f:
    data = f.readlines()
    for i in range(1, len(data)):
        weight.append(int(data[i].strip().split()[0]))
        length.append(int(data[i].strip().split()[1]))
f.close()


def takeSecond(elem):
    return elem[1]
def takeFirst(elem):
    return elem[0]
    
All_list=[]
diff_list=[]
for i in range(len(weight)):
    diff = weight[i]-length[i]
    All_list.append([diff, weight[i], length[i]])
    diff_list.append(diff)
 
All_list.sort(key=takeFirst,reverse=True)



count_dic = dict(Counter(diff_list))
step_cum=0
All_list_sort=[]
diff_set=set()
for i in range(len( weight )):
    
    if All_list[i][0] not in diff_set:
        diff_set.add(All_list[i][0])
        step=count_dic[ All_list[i][0] ]
        seg_list=All_list[step_cum:(step_cum+step)]
        seg_list.sort(key=takeSecond,reverse=True)
        All_list_sort +=seg_list
        step_cum += step
    else:
        None 


sumcomplete=0
completetime = 0
for i in range(len(weight)):
    completetime += All_list_sort[i][2]
    sumcomplete += All_list_sort[i][1]*completetime
    
print(sumcomplete)
    
    
    
## order by ratio
    
All_list=[]
with open("C:/Users/cuisa/Desktop/python/scheduling_jobs.txt") as f:
    data = f.readlines()
    for i in range(1, len(data)):
        weight=int(data[i].strip().split()[0])
        length=int(data[i].strip().split()[1])
        All_list.append([weight,length, weight/length])
f.close()



 
All_list.sort(key=lambda x:x[2],reverse=True)    
    
sumcomplete=0
completetime = 0
for i in range(len(All_list)):
    completetime += All_list[i][1]
    sumcomplete += All_list[i][0]*completetime
    
print(sumcomplete)    
    
    
    
    
    
    

## other people good code ##
    
   ## order by difference ## 
 
jobsFile = open('C:/Users/cuisa/Desktop/python/scheduling_jobs.txt','r')
lines = jobsFile.readlines()[1:]

jobs = []
length,weight = 0,0

for line in lines:
    weight = int(line.split()[0])
    length = int(line.split()[1])
    jobs.append([weight,length,weight - length])

jobs = sorted(jobs,key = lambda x:(x[2],x[0]))
jobs = jobs[::-1]#inverse, decreasing order
sumTime = 0
sumLength = 0 
for job in jobs:
    sumLength += job[1]
    sumTime += job[0] * sumLength
print(sumTime)    


  ## order by ratio ##
  
jobsFile = open('C:/Users/cuisa/Desktop/python/scheduling_jobs.txt','r')
lines = jobsFile.readlines()[1:]

jobs = []
for line in lines:
    weight = int(line.split()[0])
    length = int(line.split()[1])
    jobs.append([weight,length,float(weight) / float(length)])

jobs = sorted(jobs,key = lambda x:x[2])
jobs = jobs[-1::-1]
sumTime = 0
sumLength = 0 
for job in jobs:
    sumLength += job[1]
    sumTime += job[0] * sumLength
print(sumTime)



# Prim's algorithm


inFile = open('C:/Users/cuisa/Desktop/python/Prim.txt', 'r')

edges = []
cv = 0 # current vertex
numbers = False
included_nodes = []
tree = []
overall_cost = 0

for f in inFile:
    if(numbers == False):
        num_nodes, num_edges = map(int, f.split())
        numbers = (num_nodes, num_edges)
    else:
        node1, node2, cost = map(int, f.split())
        edges.append([node1, node2, cost])

# sorting edges by increasing order of edge cost
edges = sorted(edges, key=lambda x: x[2])

#initializing current vertex
cv = edges[0][0]
T=[]

while len(included_nodes) < numbers[0]:
    if cv not in included_nodes:
        included_nodes.append(cv)
        for e in edges:
            if ((e[0] == cv and e[1] not in included_nodes) or
                    (e[0] in included_nodes and e[1] not in included_nodes)):
                overall_cost += e[2]
                T.append(e[2])
                cv = e[1]
                break
            elif ((e[1] == cv and e[0] not in included_nodes) or
                    (e[1] in included_nodes and e[0] not in included_nodes)):
                overall_cost += e[2]
                T.append(e[2])
                cv = e[0]
                break
            else: 
                continue
       
        
print(overall_cost)






inFile = open('C:/Users/cuisa/Desktop/python/Prim.txt', 'r')
data=inFile.readlines()
graph={}
for line in data[1:]:
    u=int(line.split()[0])
    v=int(line.split()[1])
    cost=int(line.split()[2])
    if u not in graph.keys():
        graph[u]=[]
        graph[u].append([v,cost])
        if v in graph.keys():
            graph[v].append([u,cost])
        else:
            graph[v]=[]
            graph[v].append([u,cost])
    else:
        graph[u].append([v,cost])
        if v in graph.keys():
            graph[v].append([u,cost])
        else:
            graph[v]=[]
            graph[v].append([u,cost])


X={152}
head=152
V=set(graph.keys())
T1=[]


while X != V:
    global head
    list1 = list(graph[head])
    for item1 in list1:
        if item1[0] in X:
            graph[head].remove(item1)
    node_add = min(graph[head], key = lambda x:x[1])[0]
    T1.append(min(graph[head], key = lambda x:x[1])[1]) 
    T2=min(graph[head], key = lambda x:x[1])[1]
    graph[head].remove( min(graph[head], key = lambda x:x[1])  )
    for item2 in graph[node_add]:
        if item2[0] in X:
            graph[node_add].remove(item2)
    graph[head] = graph[head]+graph[node_add]
    del graph[node_add]
    X.add(node_add)










        
            
        
