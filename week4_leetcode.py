# 1. Course Schedule
def canFinish(numCourses, prerequisites):

    graph = [[] for _ in range(numCourses)]
    visit = [0 for _ in range(numCourses)]
    
    for i, j in prerequisites:
        graph[i].append(j)
        
    def dfs(i):
        if visit[i] == 1:
            return False
        if visit[i] == -1:
            return True
        visit[i] = 1
        
        for node in graph[i]:
            if not dfs(node):
                return False
        visit[i] = -1
        return True
    
    for i in range(numCourses):
        if not dfs(i):
            return False

    return True

prerequisites=[[4,3],[3,2],[2,1],[1,0],[4,1]]
numCourses=5

canFinish(numCourses,prerequisites)




# 2. Trie
class Trie(object):
    def __init__(self):
        self.child = {}
        
    def insert(self, word):
        current = self.child
        for l in word:
            if l not in current:
                current[l] = {}
            current = current[l]
        current['#']=1
    
    def search(self, word):
        current = self.child
        for l in word:
            if l not in current:
                return False
            current = current[l]
        return '#' in current
    
    def startsWith(self, prefix):
        current = self.child
        for l in prefix:
            if l not in current:
                return False
            current = current[l]
        return True


    
# 3. Coin Change (dynamic programming)

class Solution(object):
    def coinChange(self, coins, amount):

        rs = [amount+1] * (amount+1)
        rs[0] = 0
        for i in range(1, amount+1):
            for c in coins:
                if i >= c:
                    rs[i] = min(rs[i], rs[i-c] + 1)

        if rs[amount] == amount+1:
            return -1
        return rs[amount]
        
coins=[1,2,5]
amount =11

ob1=Solution()
ob1.coinChange(coins, amount)








# 4.Product of Array Except Self

class Solution:
    def productExceptSelf(self, nums):
        length = len(nums)
        L, R, answer = [0]*length, [0]*length, [0]*length
        L[0] = 1
        for i in range(1, length):

            L[i] = nums[i - 1] * L[i - 1]
        

        R[length - 1] = 1
        for i in reversed(range(length - 1)):

            R[i] = nums[i + 1] * R[i + 1]
        

        for i in range(length):
            answer[i] = L[i] * R[i]
        
        return answer
    
ob1=Solution()
nums=[2,3,4,5]
ob1.productExceptSelf(nums)



# 5. Validate Binary Search Tree

import math
class Solution(object):
    def isValidBST(self, root):
        
        def validate(node, low=-math.inf, high=math.inf):
            if not node:
                return True
            if node.val<=low or node.val>=high:
                return False
            
            return (validate(node.right, node.val, high)) and (validate(node.left, low, node.val))
        
        return validate(root)
    
    

# 6. Number of Islands
class Solution(object):
    def numIslands(self, grid):
        def dfs(grid, r,  c, R, C):
            grid[r][c] = "0"
            
            if r-1 >=0 and grid[r-1][c] == "1":
                dfs(grid, r-1, c, R, C)
            if r+1 <R and grid[r+1][c] =="1":
                dfs(grid, r+1, c, R, C)
            if c-1>=0 and grid[r][c-1] == "1":
                dfs(grid, r, c-1, R, C)
            if c+1 <C and grid[r][c+1] =="1":
                dfs(grid, r, c+1, R,C)
        
        
        
        
        R = len(grid)
        C = len(grid[0])
        num_islands = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c]=="1":
                    num_islands +=1
                    dfs(grid, r, c, R, C)
        return num_islands
                    

grid=[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
ob1=Solution()
ob1.numIslands(grid)




# 7. Rotting Oranges

# (BFS and keep tracking of coordinates of rotted oranges and keep tracking of number of fresh oranges)

class Solution(object):
    def orangesRotting(self, grid):
        position = []
        fresh_oranges = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 2:
                    grid[r][c]=3
                    position.append([r,c])
                elif grid[r][c] == 1:
                    fresh_oranges += 1

        time = 0
        position2=[]
        
        while position:
            fresh_oranges_before = fresh_oranges
            for coord in position:
                r, c = coord
                if r-1 >=0 and grid[r-1][c] == 1:
                    grid[r-1][c]=3
                    fresh_oranges -=1
                    position2.append([r-1,c])
                if r+1< len(grid) and grid[r+1][c] == 1:
                    grid[r+1][c]=3
                    fresh_oranges -=1
                    position2.append([r+1,c])                
                if c-1 >= 0 and grid[r][c-1] == 1:
                    grid[r][c-1]=3
                    fresh_oranges -=1
                    position2.append([r,c-1])
                if c+1 < len(grid[0]) and grid[r][c+1] == 1:
                    grid[r][c+1]=3
                    fresh_oranges -=1
                    position2.append([r,c+1])
                
            if fresh_oranges < fresh_oranges_before:
                    time +=1
            else:
                    None
            position = position2
            position2=[]
        return time if fresh_oranges ==0 else -1
        
            
                      
                


grid=[[0,2]]
ob1= Solution()
ob1.orangesRotting(grid) 





# 8. Search in Rotated Sorted Array (3 binary searches)

class Solution(object):
    def search(self, nums, target):
        
        def binarysearch(numlist,target):
                position=-1
  
                left=0
                right=len(numlist)-1
                
                while left<=right:
                    pivot = left+(right-left)//2
                    if numlist[pivot]>target:
                        right = pivot-1
                    elif numlist[pivot]<target:
                        left = pivot +1
                    elif numlist[pivot]==target:
                        position = pivot
                        break
                return position
            
        def findrotation(numlist):
            left= 0
            right = len(numlist)-1
                
            while left<=right:
                pivot = left+(right-left)//2
                if numlist[pivot] >=numlist[0]:
                    left= pivot+1
                else:
                    right=pivot-1
            return left
        
        turn_point = findrotation(nums)
        left_part=nums[:turn_point]
        right_part=nums[turn_point:]
            
        if not left_part or not right_part:
            return binarysearch(nums, target)
        
        if target>=nums[0]:
            return binarysearch(left_part, target)
        elif target<nums[0]:
            return binarysearch(right_part, target)+len(left_part) if binarysearch(right_part, target)!=-1 else -1









nums = [4,5,6,7,0,1,2]
target = 3
ob1= Solution()
ob1.search(nums, target)
