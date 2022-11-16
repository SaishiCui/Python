from collections import deque
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1. Serialize and Deserialize Binary Tree


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        q = deque([root])
        res = []
        
        while q:
            el = q.popleft()

            if el:
                res.append(el.val)
                q.extend([el.left,el.right])
            else:
                res.append('null')
        
        return ','.join(str(e) for e in res)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(',')
        
        if data[0]=='null':
            return None
        
        root = TreeNode(int(data[0]))
        q = deque([root])
        idx = 1
        
        def getNode(idx):
            if idx >=len(data):
                return None
            else:
                 return None if data[idx] == "null" else TreeNode(int(data[idx]))
        
        while q:
            el = q.popleft()
            
            leftNode = getNode(idx)
            if leftNode:
                el.left = leftNode
                q.append(el.left)
            idx+=1       
            
            rightNode = getNode(idx)
            if rightNode:
                el.right = rightNode
                q.append(el.right)
            idx+=1 
        
        return root


root = TreeNode(1, left = TreeNode(2, left=TreeNode(3), right=TreeNode(4)), right = TreeNode(5) )



# 2. Trapping Rain Water (dynamic programming)


class Solution(object):
    def trap(self, height):
        
        left_to_right = [0]*len(height)
        current_max = height[0]
        for i in range(len(height)):
            if height[i]>=current_max:
                left_to_right[i] = height[i]
                current_max = height[i]
            else:
                left_to_right[i] = current_max
        right_to_left = [0]*len(height)
        current_max = height[-1]
        for i in range(len(height)-1, -1 ,-1):
            if height[i]>=current_max:
                right_to_left[i] = height[i]
                current_max = height[i]
            else:
                right_to_left[i] = current_max
        ans = 0
        for i in range(len(height)):
            ans += min(left_to_right[i], right_to_left[i]  ) - height[i]
        return ans


height = [0,1,0,2,1,0,1,3,2,1,2,1]
height = [4,2,0,3,2,5]
height = [4]
ob1=Solution()
ob1.trap(height)
    



## Find Median from Data Stream
## 这道题可以归类为heap，或者优先队列问题
## 构造两个heap， 一个heap用来存储大的那一半，另一个heap用来存储小的那一半
## 主要，大的那一半直接存储，小的那一半，用负值来存储，因为小的那一半要找最大值
## 中间有调整，如果一个heap比另一个heap长2个单位， 那么需要移动


import heapq
class MedianFinder(object):

    def __init__(self):
        self.lowerhalf = [] # store the small half, top is the largest in the small part
        self.upperhalf = [] # store the large half, top is the smallest in the large part

        
    def addNum(self, num):
        if len(self.upperhalf) == 0:
            heapq.heappush(self.upperhalf, num)
            return
        
        if num >= self.upperhalf[0]:
            heapq.heappush(self.upperhalf, num)
        else:
            heapq.heappush(self.lowerhalf, -num)
            
        if len(self.upperhalf) - len(self.lowerhalf) ==2:
            transfer = heapq.heappop(self.upperhalf)
            heapq.heappush(self.lowerhalf, -transfer)
        elif len(self.lowerhalf) - len(self.upperhalf) ==2:
            transfer = heapq.heappop(self.lowerhalf)
            heapq.heappush(self.upperhalf, -transfer)
        
        

    def findMedian(self):
        if len(self.upperhalf) == len(self.lowerhalf):
            mid1 = self.upperhalf[0]
            mid2 = -self.lowerhalf[0]
            return (mid1+mid2)/2
        elif len(self.upperhalf) > len(self.lowerhalf):
            return self.upperhalf[0]
        elif len(self.lowerhalf) > len(self.upperhalf):
            return -self.lowerhalf[0]
        
        
        
ob2 = MedianFinder()
ob2.addNum(1)           
ob2.addNum(2)



ob2.findMedian()