
## 1.Huffman coding


inFile = open('C:/Users/cuisa/Desktop/python/Huffman.txt', 'r')
data=inFile.readlines()
num_list =[]
for line in data[1:]:
    num_list.append(int(line.strip()))
    
    
length=0
track = max(num_list)
while len(num_list)>1:
    num_list.sort()
    if track ==num_list[0] or track == num_list[1]:
        track=num_list[0]+num_list[1]
        length +=1
    
    next_combine = num_list[0]+num_list[1]
    num_list.pop(0)
    num_list.pop(0)
    num_list.append(next_combine)
    

print(length)




### Second solution

import heapq

class Tree(object):
    def __init__(self):
        self.minL = -1
        self.maxL = -1

with  open('C:/Users/cuisa/Desktop/python/Huffman.txt', 'r') as f:
    content = f.readlines()

numOfSymbols = int(content.pop(0))
weights = []
for l in content:
    leaf = Tree()
    leaf.minL = 0
    leaf.maxL = 0
    heapq.heappush(weights, (int(l), leaf))

while len(weights) > 1:
    w1, v1 = heapq.heappop(weights)
    w2, v2 = heapq.heappop(weights)
    node = Tree()
    node.minL = min(v1.minL, v2.minL) + 1
    node.maxL = max(v1.maxL, v2.maxL) + 1
    heapq.heappush(weights, (w1+ w2, node))

totalW, head = heapq.heappop(weights)
print(head.minL)
print(head.maxL)



## 2. weight independent set of a path graph


inFile = open('C:/Users/cuisa/Desktop/python/WIS.txt', 'r')
data=inFile.readlines()
weight_list =[]
for line in data[1:]:
    weight_list.append(int(line.strip()))

def WISValue(weight_list):
    
    dp=[0]*(len(weight_list)+1)
    dp[1]=weight_list[0]
    
    for i in range(2, len(weight_list)+1):
        dp[i]=max(dp[i-1], dp[i-2]+weight_list[i-1])
        
    dp = dp[1:]
    return dp

dp = WISValue(weight_list)



def WISSet(dp,weight_list):
    S=[]
    
    i=len(dp)-1
    while i>=0:
        if dp[i-1]>=dp[i-2]+weight_list[i]:
            i -=1
        else:
            S.append(i)
            i -=2
    return S

S= WISSet(dp,weight_list)



sol = ""

def is_in(S, v):
    if  v in S:
        return "1"
    else:
        return  "0"

nodes =  [1, 2, 3, 4, 17, 117, 517, 997]


for n in nodes:
    sol += is_in(S, n-1)  
    
    
