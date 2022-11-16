# 1. Insert Interval

def insert(intervals, newInterval):
    output = []
    idx = 0
    while idx < len(intervals) and newInterval[0]>intervals[idx][0]:
        output.append(intervals[idx])
        idx +=1
    
    if not output or output[-1][1]<newInterval[0]:
        output.append(newInterval)
    else:
        output[-1][1] =max(output[-1][1],newInterval[1])
    
    while idx< len(intervals):
        if output[-1][1]<intervals[idx][0]:
            output.append(intervals[idx])
        else:
            output[-1][1] = max(output[-1][1], intervals[idx][1])
        idx +=1
    return output
    
    
       
    
intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
newInterval = [4,8]
insert(intervals, newInterval)



# 2. 01 Matrix

def updateMatrix(matrix):

    # approach: dynamic programming to scan matrix twice from left-top and right-bottom

    m = len(matrix)
    n = len(matrix[0]) # it's promised there will be at least one zero

    # scan from left-top
    for i, row in enumerate(matrix):
        for j, ele in enumerate(row):
            if ele:
                top = matrix[i - 1][j] + 1 if i > 0 else float('inf')
                left = matrix[i][j - 1] + 1 if j > 0 else float('inf')
                matrix[i][j] = min(top, left)

    # scan from right-bottom
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            ele = matrix[i][j]
            if ele:
                bottom = matrix[i + 1][j] + 1 if i < m - 1 else float('inf')
                right = matrix[i][j + 1] + 1 if j < n - 1 else float('inf')
                matrix[i][j] = min(ele, bottom, right)

    return matrix


mat = [[0,0,0],[0,1,0],[1,1,1]]
updateMatrix(mat)



# 3. K Closest Points to Origin
class Solution:
    def kClosest(self, points,  k):
        # Sort the list with a custom comparator function
        points.sort(key=self.squared_distance)
        
        # Return the first k elements of the sorted list
        return points[:k]
    
    def squared_distance(self, point):
        
        return point[0] ** 2 + point[1] ** 2
        

points = [[3,3],[5,-1],[-2,4]]
k = 2
obj=Solution()
obj.kClosest(points, k)


import heapq


class Solution:
    def kClosest(self, points, k):
        # Since heap is sorted in increasing order,
        # negate the distance to simulate max heap
        # and fill the heap with the first k elements of points
        heap = [(-self.squared_distance(points[i]), i) for i in range(k)]
        heapq.heapify(heap)
        for i in range(k, len(points)):
            dist = -self.squared_distance(points[i])
            if dist > heap[0][0]:
                # If this point is closer than the kth farthest,
                # discard the farthest point and add this one
                heapq.heappushpop(heap, (dist, i))
        
        # Return all points stored in the max heap
        return [points[i] for (_, i) in heap]
    
    def squared_distance(self, point):
        """Calculate and return the squared Euclidean distance."""
        return point[0] ** 2 + point[1] ** 2
    
points = [[3,3],[5,-1],[-2,4]]
k = 2
obj=Solution()
obj.kClosest(points, k)



# 4. Longest Substring Without Repeating Characters 

def lengthOfLongestSubstring(s):
    out=0
    left=0
    mp={}
    for j in range(len(s)):
        if s[j] in mp:
            left = max(mp[s[j]],left)
        out = max(out, j-left+1)
        mp[s[j]]=j+1
    return out

s="xkppwker"
s = "abcabcbb"
s="dvdf"
lengthOfLongestSubstring(s)



# 5. Three sum


class Solution:                
    def threeSum(self, nums):
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i == 0 or nums[i - 1] != nums[i]:
                self.twoSumII(nums, i, res)
        return res

    def twoSumII(self, nums, i, res):
        lo, hi = i + 1, len(nums) - 1
        while (lo < hi):
            sum = nums[i] + nums[lo] + nums[hi]
            if sum < 0:
                lo += 1
            elif sum > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                    
nums = [-1,0,1,2,-1,-4]
ob1=Solution()
ob1.threeSum(nums)



class Solution1:                
    def threeSum(self, nums):
        res = []
        nums.sort()
        for i in range(len(nums)):
            if i==0:
                number = nums[i]
                target = -number
                num_list = nums[i+1:]
                res +=self.twoSum(num_list, target, number)
            else:
                None
            
            if i>0 and nums[i] != nums[i-1]:
                number = nums[i]
                target = -number
                num_list = nums[i+1:]
                res +=self.twoSum(num_list, target, number)
            else:
                None
        return res

    def twoSum(self, num_list, target, number):
        output_list=[]
        ht1={}
        ht2={}
        for i in range(len(num_list)):      
            if target-num_list[i] in ht1:
                if num_list[i] not in ht2 and num_list[i] not in ht1:
                    output_list.append([ ht1[target-num_list[i]], num_list[i], number ])
                    ht2[num_list[i]]=num_list[i]
                else:
                    None
            else:    
                ht1[num_list[i]]=num_list[i]
        return output_list



nums = [-1,0,1,2,-1,-4]
ob1=Solution1()
ob1.threeSum(nums)



# 6.  Binary Tree Level Order Traversal

def levelOrder(root):
    levels =[]
    if not root:
        return levels
    
    def helper(node, level):
        if len(levels)==level:
            levels.append([])
        
        levels[level].append(node.val)
        
        if node.left:
            helper(node.left,level+1)
        if node.right:
            helper(node,level+1)
            
    helper(root,0) 
    return levels


# 7. Evaluate Reverse Polish Notation
class Solution(object):
    def evalRPN(self, tokens):

        operations={"+": lambda a,b:a+b,
                    "-": lambda a,b:a-b,
                    "*": lambda a,b:a*b,
                    "/": lambda a,b:int(a/b)}
       
        current_position = 0
        
        while len(tokens)>1:
            
            while tokens[current_position] not in operations:
                current_position +=1
                
            operator = tokens[current_position]
            number_1 = int(tokens[current_position-2])
            number_2 = int(tokens[current_position-1])
            
            operation=operations[operator]
            tokens[current_position] = operation(number_1, number_2)
            
            tokens.pop(current_position-2)
            tokens.pop(current_position-2)
            current_position -=1
        return tokens[0]
            
        
ob1=Solution()
tokens=["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
ob1.evalRPN(tokens)



# 8.  Clone Graph



class Solution(object):
    def __init__(self):
        self.visited={}
        
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """

        if not node:
            return node
        
        if node in self.visited:
            return self.visited[node]
        
        clone_node = Node(node.val, [])
        
        self.visited[node] = clone_node
        
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]
            
        return clone_node