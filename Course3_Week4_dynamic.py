import numpy as np

inFile = open('C:/Users/cuisa/Desktop/python/knapsack1.txt', 'r')
data=inFile.readlines()

Capacity=int(data[0].split()[0])
N=int(data[0].split()[1])


value =[]
size =[]
for line in data[1:]:
    value.append(int(line.split()[0]))
    size.append(int(line.split()[1]))
    
dp=np.array( [[0]*(N+1)]*(Capacity+1) )

for i in range(1,N+1):
    for x in range(Capacity+1):
        if x>=size[i-1]:
            dp[x][i] = max(dp[x][i-1], dp[x-size[i-1]][i-1]+value[i-1])
        else:
            dp[x][i] =dp[x][i-1]
            
            
            


## Hard
inFile = open('C:/Users/cuisa/Desktop/python/knapsack2.txt', 'r')
data=inFile.readlines()

Capacity=int(data[0].split()[0])
N=int(data[0].split()[1])


value =[]
size =[]
for line in data[1:]:
    value.append(int(line.split()[0]))
    size.append(int(line.split()[1]))



A_old=[0]*(Capacity+1)

for i in range(1,N+1):
    A_new = [0]*(Capacity+1)
    for j in range(Capacity+1):
        if j>=size[i-1]:
            A_new[j]=max(A_old[j], A_old[j-size[i-1]]+value[i-1]  )
        else:
            A_new[j]=A_old[j]
    A_old = A_new
    
print(A_old[Capacity])

