
## Merge k Sorted Lists
## 第一个可以用暴力算法
## 用heap可以减少运行时间 （但是要用tuple来记录idx）


import heapq
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

node1= ListNode(1, next= ListNode(4, next=ListNode(5) ))
node2= ListNode(1, next= ListNode(3, next=ListNode(4) ))
node3= ListNode(2, next= ListNode(6 ))
lists = [node1, node2, node3]
        
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next

   

    def mergeKLists2(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        h = [(l.val, idx) for idx, l in enumerate(lists) if l]
        heapq.heapify(h)
        head = cur = ListNode(None)
        while h:
            val, idx = heapq.heappop(h)
            cur.next = ListNode(val)
            cur = cur.next
            node = lists[idx] = lists[idx].next
            if node:
                heapq.heappush(h, (node.val, idx))
        return head.next


## Largest Rectangle in Histogram
## 用stack 来记录index, 如果下一个高度比上一个高，那么往stack里面加
## 如果下一个高度比上一个矮，那么开始pop，计算面积
## stack 底部是一个-1
## 如果for递归完成，stack还不是只有-1一个值，那么继续pop


class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = [-1]
        max_area = 0
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] > heights[i]:
                cur_h = heights[stack.pop()]
                cur_w = i-stack[-1]-1
                max_area = max(max_area,cur_h*cur_w)
                
            stack.append(i)
        
        while stack[-1] != -1:
            cur_h = heights[stack.pop()]
            cur_w = len(heights)-stack[-1]-1
            max_area = max(max_area, cur_h*cur_w)
        
        return max_area



heights = [2,1,5,6,2,3]
ob1 = Solution()
ob1.largestRectangleArea(heights)




## Binary Tree Maximum Path Sum
## postorder 后序遍历
## 记住一定要单边， helper的return是 node.val + max(left,right) 左右子树最大值
## 如果子树sum小于0，那么不要取该子树，所以用max(helper(node.left),0)和max(helper(node.right),0)来递归
## 用max_sum 来记录最大path sum
## 最后返回这个最大的path sum


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


import math
class Solution:
    def maxPathSum(self, root):
        
        
        max_sum = -math.inf
        
        def helper(node):
            nonlocal max_sum
            if not node:
                return 0
            
            left  = max(helper(node.left) ,0)
            right = max(helper(node.right) ,0)
            
            max_sum = max(max_sum, node.val+left+right)
            
            
            return node.val + max(left, right)
        
        helper(root)
        return max_sum
    
    
root = TreeNode(-10, left=TreeNode(9), right=TreeNode(20, left=TreeNode(15), right=TreeNode(7)  )   )
ob1 = Solution()
ob1.maxPathSum(root)


