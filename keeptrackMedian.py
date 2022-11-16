import heapq

num_list=[]
with open("C:/Users/cuisa/Desktop/Median.txt") as f:
    data = f.readlines()
    for line in data:
        num_list.append(int( line.strip()) )
f.close()




def keep_median(list):
    right=[]
    left=[]
    median=[]
    first_num  = list[0]
    second_num = list[1]
    if first_num >= second_num:
        right.append(first_num)
        left.append(second_num * -1)
    else:
        right.append(second_num)
        left.append(first_num * -1)
    heapq.heapify(left)
    heapq.heapify(right)
    median.append( first_num )
    median.append( left[0] * -1 )
    
    for number in list[2:]:
        if number > right[0]:
            heapq.heappush(right, number)
        elif number < (left[0] * -1):
            heapq.heappush(left, number * -1)
        else:
            heapq.heappush(left, number * -1)
            
        while( len(left)-len(right)>=2 ):
            move=heapq.heappop(left)
            heapq.heappush(right, move * -1)
        while( len(right)-len(left)>=2 ):
            move=heapq.heappop(right)
            heapq.heappush(left, move * -1) 
        if len(right)> len(left):
            median.append(right[0])
        if len(right)<= len(left):
            median.append(left[0]* -1 )
    return(median)

            
median_list=keep_median(num_list) 


print(sum(median_list)%10000)
    