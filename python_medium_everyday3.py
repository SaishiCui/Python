## Daily Temperatures
## 主要思想，用一个stack 来保存没有复原的日子， stack append 和 pop 反复用
## O(n) 因为stack 里面最多有n个


class Solution:
    def dailyTemperatures(self, temperatures):
        n = len(temperatures)
        answer = [0]*n
        stack = []
        
        for curr_day, curr_temp in enumerate(temperatures):
            
            while stack and temperatures[stack[-1] ] < curr_temp:
                prev_day = stack.pop()
                answer[prev_day] = curr_day - prev_day
            
            stack.append(curr_day)
        return answer



temperatures = [73,74,75,71,69,72,76,73]
ob1= Solution()
ob1.dailyTemperatures(temperatures)




## House Robber
## 主要考察动态编程， 递推公式，和base case 

class Solution:
    def rob(self, nums):
        dp = [0]*(len(nums)+1)
        dp[0]=0
        dp[1]=nums[0]
        
        for i in range(2,len(dp)):
            dp[i] = max( dp[i-1], dp[i-2]+nums[i-1])
        
        return dp[-1]
    
    
## Next Permutation
## 注意，原位移动，不能用新的数组
## 思路就是，先从后往前check, 如果是逆序，那么直接掉头，返回
## 如果找到了一个升序，那么记录下这个位置备用, 记为i
## 再从后往这个记录下来的位置check, 找到第一个大于该位置的元素的位置，记为j
## 替换i和j
## 再把i后面的部分全部调转方向

s = "abcde"
a=range(len(s)-1,-1,-1)

def nextPermutation(self, nums):
    i = j = len(nums)-1
    while i > 0 and nums[i-1] >= nums[i]:
        i -= 1
    if i == 0:   # nums are in descending order
        nums.reverse()
        return 
    k = i - 1    # find the last "ascending" position
    while nums[j] <= nums[k]:
        j -= 1
    nums[k], nums[j] = nums[j], nums[k]  
    l, r = k+1, len(nums)-1  # reverse the second part
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l +=1 
        r -= 1


nums = [1,5,8,4,7,6,5,3,1]
nums = [1,1,5]
nums = [1,2,3]
nums = [4,3,2,1]
ob1=Solution()
ob1.nextPermutation(nums)
nums




## Gas Station
## 贪心算法，只要总共加的油大于cost，那么一定有一个位置可以进行轮回
## 从0开始，如果到j点，发现current tank 的油小于0， 那么从0到j任何一点都不能作为最终的point
## 只能从j的下一个开始，因此只需要O(N)


class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        L = len(gas)
        total_tank = 0
        curr_tank =0
        start_station = 0
        
        for i in range(L):
            total_tank += gas[i] - cost[i]
            curr_tank += gas[i]-cost[i]
            
            if curr_tank<0:
                start_station = i+1
                curr_tank = 0
        
        return start_station if total_tank>=0 else -1

gas = [1,2,3,4,5]
cost = [3,4,5,1,2]




## Maximum Product Subarray
## 动态编程
## 注意，难点在于要记录最大值和最小值
## dp[n]的意思是把第n个值放进去的话，最大能多少
## dp[n] = max( 最大最小dp[n-1], nums[n] )

class Solution:
    def maxProduct(self, nums):
        n = len(nums)
        max_record = [0]*n
        min_record = [0]*n
        min_record[0] = nums[0]
        max_record[0] = nums[0]
        result = nums[0]
        
        for i in range(1,n):
            min_record[i] = min(nums[i], min_record[i-1]*nums[i], max_record[i-1]*nums[i]  )
            max_record[i] = max(nums[i], min_record[i-1]*nums[i], max_record[i-1]*nums[i]  )
            result = max(result, max_record[i])
            
        
        return result
    
nums=[2,3,-2,4,1,-2,-1]




## Design Add and Search Words Data Structure
## 前缀 Trie
## 构造的时候，遇到在当前{}下没遇到过的字母，加上{}
## 然后把当前位置移动到下一个{}里面，知道for完成，最里面放一个$ 代表能找到这个单词

## search 的时候主要注意“.” 如果遇到这个点，要遍历当前node中所有的点的子树。


class WordDictionary(object):

    def __init__(self):
        self.trie = {}
        

    def addWord(self, word):
        node = self.trie
        for ch in word:
            if not ch in node:
                node[ch] = {}
                node = node[ch]
            else:
                node = node[ch]
            
            
        node["$"]=True
        
    
    def search(self, word):
        
        def search_in_node(word, node):
            for i, ch in enumerate(word):
                if ch in node:
                    node = node[ch]
                else:
                    if ch != ".":
                        return False 
                    else:
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]):
                                return True
                        return False
    
            return "$" in node
        
        return search_in_node(word, self.trie)
                
        




ob1 = WordDictionary()

ob1.addWord("abc")
ob1.addWord("abd")
ob1.addWord("abfk")

ob1.addWord("def")
ob1.addWord("ghi")
ob1.addWord("k")
ob1.trie
ob1.search("ab.k")


## Pacific Atlantic Water Flow
## 边界问题之王
## 采用BFS搜索
## 首先要确定从哪开始递归，Pacific和Atlantic各有一行一列
## 最后取交集

from collections import deque
class Solution:
    def pacificAtlantic(self, matrix):
        if not matrix or not matrix[0]:
            return []
        
        R, C = len(matrix), len(matrix[0])
        
        pacific_queue = deque()
        atlantic_queue = deque()
        
        for i in range(R):
            pacific_queue.append((i,0))
            atlantic_queue.append((i, C-1)  )
        
        for i in range(C):
            pacific_queue.append((0,i))
            atlantic_queue.append( (R-1,i) )
        
        
        def bfs(queue):
            reachable = set()
            while queue:
                (r,c) = queue.popleft()
                reachable.add((r,c))
                for (x,y) in [(1,0),(0,1),(-1,0),(0,-1)]:
                    new_r, new_c = r+x, c+y
                    
                    if new_r<0 or new_r>=R or new_c < 0 or new_c>=C:
                        continue
                    if (new_r, new_c) in reachable:
                        continue
                    if matrix[new_r][new_c] < matrix[r][c]:
                        continue 
                    queue.append((new_r,new_c))
            return reachable
        
        pacific_reachable = bfs(pacific_queue)
        atlantic_reachable = bfs(atlantic_queue)
        
        return list(pacific_reachable.intersection(atlantic_reachable))
                




matrix = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
ob1 = Solution()
ob1.pacificAtlantic(matrix)



## Remove Nth Node From End of List
## 我的方法要遍历两次，人家只需要一次，用快慢指针

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
        
class Solution:
    def removeNthFromEnd(self, head, n):
        node = head 
        count = 0
        while node:
            node=node.next
            count +=1
        position = count - n-1
        
        if position <0:
            head = head.next
            return head
        
        node2= head
        for i in range(position):
            node2= node2.next 
        node2.next = node2.next.next
        
        return head
        
        

head = ListNode(1, next= ListNode(2, next=ListNode(3, next= ListNode(4, next=ListNode(5)))))
head =ListNode(1, next=ListNode(2))
head= ListNode(1)
n=1
n=4      

ob1 = Solution()
a=ob1.removeNthFromEnd(head, n)




## 双指针秒杀链表
## 快慢指针最开始都是指向head头部，快指p1针先走n步， 然后慢指针和快指针一起走，直到快指针none
## 此时慢指针指的就是要被删除的node
## 然后利用这个函数找出上一个node 进行操作

class Solution(object):
    def removeNthFromEnd(self, head, n):
        
        def findFromEnd(node,k):
            
            p1 = node
            
            for i in range(k):
                p1 =p1.next
                
            p2 = node
            
            while p1:
                p1 = p1.next
                p2 = p2.next
            
            return p2
        
        
        dummy = ListNode(-1)
        dummy.next = head
        x= findFromEnd(dummy, n+1)
        x.next = x.next.next
        return dummy.next 
    
    
    

## Find the Duplicate Number
## 龟兔赛跑
## 2 phases
## 1st phase to find intersection point
## 2nd phase to find the duplicate point

class Solution:
    def findDuplicate(self, nums):
        t = nums[0]
        h = nums[0]
        
        while True:
            t = nums[t]
            h = nums[nums[h]]
            if t == h:
                break
        
        t=nums[0]
        
        while t!=h:
            t = nums[t]
            h = nums[h]
        
        return h
    
ob1 = Solution()
nums = [1,3,2,4,5,4]
ob1.findDuplicate(nums)



## Shortest Path to Get Food
## BFS 强边际问题
## BFS主要注意点在于，要设定一个queue, 如果要记录step, 要用level order 

class Solution(object):
    def getFood(self, grid):
        direction = [(1,0), (0,1), (-1,0), (0,-1)]
        R = len(grid)
        C = len(grid[0])
        
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == "*":
                    start = (r,c)
                    break
        
        q = deque([start])
        res = 0
        reachable =set()
        
        while q:
            size = len(q)
            
            for _ in range(size):                
                r,c = q.popleft()
                reachable.add((r,c))
                
                if grid[r][c] == "#":
                    return res
                
                for (x,y) in direction:
                    new_r, new_c = r+x, c+y
                    if 0<=new_r<R and 0<=new_c<C and grid[new_r][new_c] != "X" and (new_r, new_c) not in reachable:
                        q.append((new_r,new_c))
                        reachable.add((new_r,new_c))
            res +=1
        
        return -1 
            
                    
                
                


grid = [["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]]
ob1 =Solution()
ob1.getFood(grid) 



## Top K Frequent Words
## 用一个heap来提升速度

import heapq
from collections import Counter

class Solution:
    def topKFrequent(self, words, k):
        count =Counter(words)
        heap = (heapq.nsmallest(k, count.items(), key= lambda item: (-item[1], item[0])))
        return [word for word, _ in heap]
    

words = ["the","day","is","sunny","the","the","the","sunny","is","is"]
words = ["i","love","leetcode","i","love","coding"]
ob1 = Solution()
ob1.topKFrequent(words, 4)




## Longest Increasing Subsequence
## 动态编程，O（N^2）
## 二分法不好想， O(NlogN)


class Solution:
    def lengthOfLIS(self, nums):
        dp = [1]*len(nums)

        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j]>=nums[i]:
                    None
                else:
                    dp[i] = max(dp[i], dp[j]+1)
                
        return max(dp)
        




nums = [10,9,2,5,3,7,101,18]
nums = [0,1,0,3,2,3]
nums = [7,7,7,7,7,7,7]
ob1 = Solution()
ob1.lengthOfLIS(nums)

 ## 扑克牌二分法
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n= len(nums)
        top = [0]*n
        piles = 0
        for i in range(n):
            poker = nums[i]
            left = 0
            right = piles
            
            while left<right:
                mid = left+(right-left)//2
                if top[mid] >= poker:
                    right = mid
                elif top[mid] < poker:
                    left = mid +1

            
            if left == piles:
                piles +=1
                
            top[left] = poker
        
        return piles
        
    
    
    
## Graph Valid Tree
## DFS 要注意， adjacent list 的构建 
## 还要注意，要记录parent 

def validTree(self, n, edges):
    
    if len(edges) != n - 1:
        return False
    
    adj_list = [[] for _ in range(n)]
    for A, B in edges:
        adj_list[A].append(B)
        adj_list[B].append(A)
    
    parent = {0: -1}
    stack = [0]
    
    while stack:
        node = stack.pop()
        for neighbour in adj_list[node]:
            if neighbour == parent[node]:
                continue
            if neighbour in parent:
                return False 
            parent[neighbour] = node
            stack.append(neighbour)
    
    return len(parent) == n

        
            



edges = [[0,1],[2,3]]
n = 3
edges=[[1,0],[2,0]]
n=5
edges = [[0,1],[0,4],[1,4],[2,3]]
n=4
edges = [[2,3],[1,2],[1,3]]
ob1= Solution()
ob1.validTree(n, edges)



## Course Schedule II


from collections import defaultdict
class Solution:

    WHITE = 1
    GRAY = 2
    BLACK = 3

    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        # Create the adjacency list representation of the graph
        adj_list = defaultdict(list)

        # A pair [a, b] in the input represents edge from b --> a
        for dest, src in prerequisites:
            adj_list[src].append(dest)

        topological_sorted_order = []
        is_possible = True

        # By default all vertces are WHITE
        color = {k: Solution.WHITE for k in range(numCourses)}
        def dfs(node):
            nonlocal is_possible

            # Don't recurse further if we found a cycle already
            if not is_possible:
                return

            # Start the recursion
            color[node] = Solution.GRAY

            # Traverse on neighboring vertices
            if node in adj_list:
                for neighbor in adj_list[node]:
                    if color[neighbor] == Solution.WHITE:
                        dfs(neighbor)
                    elif color[neighbor] == Solution.GRAY:
                         # An edge to a GRAY vertex represents a cycle
                        is_possible = False

            # Recursion ends. We mark it as black
            color[node] = Solution.BLACK
            topological_sorted_order.append(node)

        for vertex in range(numCourses):
            # If the node is unprocessed, then call dfs on it.
            if color[vertex] == Solution.WHITE:
                dfs(vertex)

        return topological_sorted_order[::-1] if is_possible else []



numCourses = 4
prerequisites = [[1,0],[2,0],[3,1],[3,2]]





## Swap Nodes in Pairs
## recursive 递归
## 两个两个递归，当最后一个是NULL或者最后一个的next是NULL的话， 返还head
## Swap 两个node, 最后返回第二个node 


class Solution(object):
    def swapPairs(self, head):
        
        if not head or not head.next:
            return head
        
        # Nodes need to be swapped
        
        first_node = head
        second_node = head.next 
        
        
        # Swapping
        first_node.next = self.swapPairs(second_node.next)
        second_node.next = first_node
        
        # Now the head is the second node 
        return second_node



## Path Sum II
## Given the root of a binary tree and an integer targetSum,
## return all root-to-leaf paths where the sum of the node values in the path equals targetSum. 
## Each path should be returned as a list of the node values, not node references.

## Backtrack 思路， 要维护

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def pathSum(self, root, targetSum):
        pathsList = []
        self.helper(root, targetSum, [], pathsList)
        
        return pathsList
    
    def helper(self, node, remainingSum, pathNodes, pathsList):
        
        if not node:
            return
        
        pathNodes.append(node.val)
        
        if remainingSum == node.val and not node.left and not node.right:
            pathsList.append(list(pathNodes))
        else:
            self.helper(node.left, remainingSum - node.val, pathNodes, pathsList)
            self.helper(node.right, remainingSum - node.val, pathNodes, pathsList)
        
        ## 维护
        pathNodes.pop()
        
root = TreeNode(5, left= TreeNode(4, left=TreeNode(11, left=TreeNode(7), right=TreeNode(2)  )), right=TreeNode(8, left=TreeNode(13), right=TreeNode(4, left=TreeNode(5), right = TreeNode(1) )))
        
ob1 = Solution()
ob1.pathSum(root, 22)
        


## Longest Consecutive Sequence
## Given an unsorted array of integers nums, 
## return the length of the longest consecutive elements sequence.
## nums = [100,4,200,1,3,2] output = 4 ([1,2,3,4])

## 把numbers先放到set中
## 倒着来，如果一个number减1在set中，先不用管它
## 如果一个number减1不在set中，那么管它，每次加1，看看新的加1的整数在不在set里面，在的话，计数加1


class Solution:
    def longestConsecutive(self, nums):
        
        longest_streak = 0
        num_set = set(nums)
        
        for num in num_set:
            if num - 1 not in num_set:
                current_streak = 1
                current_num = num
                
                while current_num + 1 in num_set:
                    current_streak +=1
                    current_num +=1
                
                longest_streak = max(longest_streak, current_streak)
                
        return longest_streak

ob1 = Solution()
nums = [1,2,3,4,9,10,11]
ob1.longestConsecutive(nums)


nums = [1,2,3,4,5,6,7]
k = 30

## Rotate Array
## Given an array, rotate the array to the right by k steps, where k is non-negative.
## Input: nums = [1,2,3,4,5,6,7], k = 3
## Output: [5,6,7,1,2,3,4]

## 第一种是我自己的方法，注意k如果超出了n，那么用%
## 第二个是人家的方法，reverse list


class Solution:
    def rotate(self, nums, k):
        k %= len(nums)
        
        nums[:]=nums[-k:]+nums[:-k]



class Solution:
    def rotate(self, nums, k):
        
        n = len(nums)
        k %= n
        
        self.reverse(nums, 0, n-1)
        self.reverse(nums, 0, k-1)
        self.reverse(nums, k, n-1)
        
    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end =start +1, end -1 


















## Odd Even Linked List
## Given the head of a singly linked list, 
## group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.
## 用0开头占位， 然后用oddsHead和evensHeads打头阵， 用head循环

class Solution:
    def oddEvenList(self, head):
        odds = ListNode(0)
        evens = ListNode(0)
        oddsHead = odds
        evensHead = evens
        isOdd = True
        while head:
            if isOdd:
                odds.next = head
                odds = odds.next
            else:
                evens.next = head
                evens = evens.next
            isOdd = not isOdd
            head = head.next
        evens.next = None
        odds.next = evensHead.next
        return oddsHead.next

    

            


head = ListNode(1, next = ListNode(2, next= ListNode(3, next = ListNode(4))))









## Decode String
## 递归，注意self.i 记录i 指针
 
## 第二个方法 用stack, 记录 数字， 记录数字后面括号里面的cur_level 


s = "3[a2[c]]"
s = "2[k3[da]ef]"


        

class Solution:
    def decodeString(self, s) :
        def decode():
            result = ""
            while self.i < len(s) and s[self.i] != ']':
                if s[self.i].isdigit():
                    idx = s.index('[', self.i)
                    digit = int(s[self.i:idx])
                    self.i = idx + 1
                    result += digit * decode()
                else:
                    result += s[self.i]
                    self.i += 1
                
            self.i += 1
            return result
        
        self.i = 0
        return decode()
    
ob1=Solution()
ob1.decodeString(s)


def decodeString(self, s):
        stack = []
        cur_level = []
        num = 0
        
        for char in s:
            if char.isdigit():
                num = num * 10 + int(char)
            
            elif char.isalpha():
                cur_level.append(char)
            
            elif char == '[':
                stack.append((num, [*cur_level]))
                cur_level = []
                num = 0
            
            elif char == ']':
                prev_level_num, prev_level = stack.pop()
                cur_level_string = "".join(cur_level)
                cur_level = [*prev_level, prev_level_num * cur_level_string] 
            
        return "".join(cur_level)



s = "3[a2[c]]"
s = "2[k3[da]ef]"



##  Contiguous Array
## Given a binary array nums,
##  return the maximum length of a contiguous subarray with an equal number of 0 and 1.

## 随机游走图像， 遇到1加1个count， 遇到0加-1个count
## 用hashtable 记录第一次到达一个count值的index， 注意要与hashtable 里面的第一个值的index 0 做出区分，也就是idx要加1
## count不在table里面的话，记录这个count和所对应的index+1， count在table里的话，维持max_length

class Solution:
    def findMaxLength(self, nums):
        
        count = 0
        maxL = 0
        ht = {0:0}
        for idx, num in enumerate(nums):
            if num == 0:
                count -=1
            else:
                count +=1
                
            if count in ht:
                maxL = max(maxL, idx+1-ht[count])
            else:
                ht[count]= idx+1
        
        return maxL
    

ob1 = Solution()
nums= [0, 1]
ob1.findMaxLength(nums)



## Maximum Width of Binary Tree
## BFS 给标号 左边孩子2*目前的标记，右边孩子2*目前标记+1


class Solution:
    def widthOfBinaryTree(self, root):
        if not root:
            return 0
        
        max_width = 0
        
        queue = deque()
        queue.append( (root, 0) )
        
        while queue:
            level_length = len(queue)
            level_head_index = queue[0][1]
            
            for _ in range(level_length):
                node, col_idx = queue.popleft()
                if node.left:
                    queue.append((node.left, 2*col_idx) )
                if node.right:
                    queue.append((node.right, 2*col_idx+1))
            
            max_width = max(max_width, col_idx - level_head_index +1 )
        
        return max_width
            

root = TreeNode(4, left=TreeNode(3, left=TreeNode(1), right=TreeNode(2)) , right = TreeNode(5))        


##  Find K Closest Elements


class Solution:
    def findClosestElements(self, arr, k, x):
        
        arr_list = []
        for num in arr:
            arr_list.append([abs(num-x), num])
        
        
        sorted_arr=sorted(arr_list, key = lambda x:(x[0],x[1]))
        
        out_arr = []
        for i in range(k):
            out_arr.append(sorted_arr[i][1] )
        
        sorted_out = sorted(out_arr)
        
        return sorted_out

arr = [1,2,3,4,5]
k = 4
x = 3

arr = [1,2,3,4,5]
k = 4
x = -1
ob1 = Solution()
ob1.findClosestElements(arr, k, x)





## Inorder Successor in BST
## Given the root of a binary search tree and a node p in it, return the in-order successor of that node in the BST. 
## If the given node has no in-order successor in the tree, return null.

## 利用二叉搜索树的特性，左边的子树的value要小于右边，因此，如果p比root的value大，那么root的左侧就全部可以排除，然后root=root.right
## 如果p比root的value小，那么root的右侧可以被删除，然后root=root.left，但是需要记住，此时root可能是p的successor


class Solution:
    def inorderSuccessor(self, root, p):
        
        successor = None
        
        while root:
            if p.val >= root.val:
                root = root.right
            else:
                successor = root
                root = root.left
        
        return successor
    
    
## Jump game
## You are given an integer array nums. You are initially positioned at the array's first index,
## and each element in the array represents your maximum jump length at that position.
## 跳一跳问题，其实就是看最多能跳多远，动态规划
## Input: nums = [2,3,1,1,4]
## Output: true
## Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.


class Solution:
    def canJump(self, nums):
        n = len(nums)
        farthest = 0
        
        for i in range(n-1):
            farthest = max(farthest, i+nums[i])
            
            ## 可能卡在0处
            if farthest <= i:
                return False
            
        return farthest >= n-1
    


##  Add Two Numbers
## You are given two non-empty linked lists representing two non-negative integers.
## The digits are stored in reverse order, 
## and each of their nodes contains a single digit. 
## Add the two numbers and return the sum as a linked list.


## Input: l1 = [2,4,3], l2 = [5,6,4]
## Output: [7,0,8]
## Explanation: 342 + 465 = 807.

class Solution:
    def addTwoNumbers(self, l1, l2):
        out_node = ListNode(-1)
        curr = out_node 
        carry = 0
        
        while l1 or l2 or carry:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            
            sum1 = l1_val + l2_val + carry 
            carry = sum1//10
            newNode = ListNode(sum1%10)
            curr.next = newNode
            curr = newNode
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return out_node.next 
    
    

## Generate Parentheses
## backtrack 回溯算法
## 回溯算法秘诀:递归之后pop


class Solution:
    def generateParenthesis(self, n):
        
        
        ans = []
        def backtrack(S = [], left = 0, right =0):
            if len(S) == 2*n:
                ans.append("".join(S))
            if left < n:
                S.append("(")
                backtrack(S, left+1, right)
                S.pop()
            if right<left:
                S.append(")")
                backtrack(S, left, right+1)
                S.pop()
        
        backtrack()
        return ans       


##  Sort List
## Given the head of a linked list, return the list after sorting it in ascending order.
class Solution:
    def sortList(self, head):

        arr = []
        helper = ListNode(0)
        helper.next = head
        
        while head:
            arr.append(head.val)
            head = head.next
        arr.sort()
        c = 0
        head2 = helper.next
        while head2:
            head2.val = arr[c]
            c+=1
            head2 = head2.next
        return helper.next 



## Number of Connected Components in an Undirected Graph

## 先写一个并查集 Uion-Find
## 并查集一共有三个元素，第一个是root，也就是parent node， 初始化是自己是自己的parent
## 第二个元素是weight， 用来优化合并
## 第三个元素是count， 用来记录connected component


class UnionFind:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.weight = [1]*n
        self.count = n
        
    def find(self, x):
        if x == self.parent[x]:
            return x
        
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xparent = self.find(x)
        yparent = self.find(y)
        
        if xparent != yparent:
            self.count -=1
            if self.weight[xparent] > self.weight[yparent]:
                self.parent[yparent] = xparent
            elif self.weight[xparent] < self.weight[yparent]:
                self.parent[xparent] = yparent
            else:
                self.parent[yparent] = xparent
                self.weight[xparent] +=1
        
class Solution():
    def countComponents(self, n, edges):
        uf = UnionFind(n)
        for edge in edges:
            uf.union(edge[0], edge[1])
        
        return uf.count


n = 5
edges = [[0,1],[1,2],[3,4]]
ob1 = Solution()
ob1.countComponents(n, edges)            
        





## Minimum Knight Moves

## bfs 

class Solution:
    def minKnightMoves(self, x, y):
        choices = [(1,2), (2,1), (-1,-2), (-2, -1),
                   (1,-2), (2,-1), (-1,2), (-2,1)]
        
        def bfs(x,y):
            steps = 0
            visit = set()
            queue = deque([(0,0)])
            
            while queue:
                curr_count = len(queue)
                for i in range(curr_count):
                    curr_x, curr_y = queue.popleft()
                    if (curr_x, curr_y) == (x,y):
                        return steps 
                    
                    for move_x, move_y in choices:
                        next_x, next_y = curr_x + move_x, curr_y + move_y
                        if (next_x, next_y) not in visit:
                            visit.add( (next_x, next_y) )
                            queue.append( (next_x, next_y ) )
                
                steps +=1
        
        return bfs(x,y)
    




ob1 = Solution()
x=2
y=1
ob1.minKnightMoves(x, y)



## Subarray Sum Equals K

## Given an array of integers nums and an integer k, 
## return the total number of subarrays whose sum equals to k.
## A subarray is a contiguous non-empty sequence of elements within an array.
## Input: nums = [1,2,3], k = 3
## Output: 2         
            
class Solution:
    def subarraySum(self, nums, k):    
        sums = 0
        count = 0
        ht = {0:1}
        
        for i in range(len(nums)):
            sums += nums[i]
            if sums-k in ht:
                count += ht[sums-k]
            ht[sums] = ht.get(sums, 0) +1
        
        return count 

nums = [1,2,3]
k=3
ob1 = Solution()
ob1.subarraySum(nums, k)



## Path Sum III
## 和上一题一样，但是需要运用preorder traversal 以及backtrack


class Solution:
    def pathSum(self, root, targetSum):
        
        def preorder(node, curr_sum):
            nonlocal count
            if not node:
                return 
            
            curr_sum += node.val
            if curr_sum - k in ht:
                count += ht[curr_sum-k]
            ht[curr_sum] = ht.get(curr_sum, 0) + 1
            
            preorder(node.left, curr_sum)
            preorder(node.right, curr_sum)
            
            ht[curr_sum] -=1
        
        count, k = 0, targetSum
        ht = {0:1}
        preorder(root, 0)
        return count 


## Pow(x, n)
## faster power 
## 遇到奇数，递归x*myPower(x,n-1)
## 遇到偶数，递归myPower(x*x, n/2)

class Solution:
    def myPow(self, x, n):
        if n == 0:
            return 1
        if x == 0:
            return 0
        if n<0:
            n = -n
            x = 1/x
        if n%2 == 0:
            return self.myPow(x*x, n/2)
        else:
            return x*self.myPow(x, n-1)

x= 2
n=10
ob1 = Solution()
ob1.myPow(x, n)





## Asteroid Collision
## Input: asteroids = [5,10,-5]
## Output: [5,10]
## 利用stack， 当stack非空的时候，遇到一个往左的（-），并且前面是一个往右的（+）,做个大小对比，往左的大的话
## 继续往左判断，往右的大的话，不用管这个新遇到的, 一样的话，pop然后下一个


class Solution(object):
    def asteroidCollision(self, asteroids):
        ans = []
        for new in asteroids:
            while ans and new<0<ans[-1]:
                if ans[-1] < -new:
                    ans.pop()
                    continue
                elif ans[-1] > -new:
                    break
                else:
                    ans.pop()
                    break
            else:
                ans.append(new)
        return ans 

ob1 = Solution()
asteroids = [5,10,-5]
ob1.asteroidCollision(asteroids)


## Random Pick with Weight

## For example, if w = [1, 3], 
## the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), 
##and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).
import random

class Solution:

    def __init__(self, w):
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum
        
    
    def pickIndex(self):
        target = self.total_sum * random.random()
        
        for i, prefix_sum in enumerate(self.prefix_sums):
            if target < prefix_sum:
                return i
            
ob1= Solution([1,9])
ob1.pickIndex()



## Maximal Square
## Given an m x n binary matrix filled with 0's and 1's, 
## find the largest square containing only 1's and return its area.

## 动态规划，二维数组，dp[i][j]代表， 以第[i][j]个元素为右下角的正方形，最大的边长
## 如果这个位置的上，左，左上，三个位置都是1，则边长可以扩大1.



class Solution:
    def maximalSquare(self,matrix):
        if not matrix:
            return 0
        
        m, n, max_len = len(matrix), len(matrix[0]), 0
        dp = [[0]*(n+1) for _ in range(m+1)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
                    max_len = max(max_len, dp[i+1][j+1])
                else:
                    dp[i+1][j+1] = 0
        
        return max_len**2
        
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
ob1 = Solution()
ob1.maximalSquare(matrix)




## Rotate Image
## 先沿着对角线折叠，再按行调转

class Solution:
    def rotate(self, matrix):

        n = len(matrix)
        for i in range(n):
            for j in range(i,n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        for i in range(n):
            left, right = 0, n-1
            while left<right:
                matrix[i][left], matrix[i][right] = matrix[i][right], matrix[i][left]
                left +=1
                right -=1
                
                
                
##  Design Hit Counter
## Design a hit counter which counts the number of hits received 
## in the past 5 minutes (i.e., the past 300 seconds).
                
## 用stack储存timestamp
## timestamp -300 然后二分搜索，确定left值 


class HitCounter:

    def __init__(self):
        self.hits = []
        

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        left, right =0, len(self.hits) - 1
        target = timestamp - 300
        
        while left<=right:
            mid = (left+right)//2
            if self.hits[mid] <= target:
                left = mid+1
            else:
                right = mid -1
        
        return len(self.hits) -left
    
    
    
## Search a 2D Matrix

## Integers in each row are sorted from left to right.
##The first integer of each row is greater than the last integer of the previous row.

## 二分搜索，主要注意 矩阵元素的定位，用//和%


class Solution:
    def searchMatrix(self, matrix, target):
        m = len(matrix)
        n = len(matrix[0])
        
        left, right = 0 , m*n -1
        
        while left<=right:
            pivot_idx = (left+right)//2
            pivot_element = matrix[pivot_idx // n][pivot_idx % n]
            if target == pivot_element:
                return True
            elif target < pivot_element:
                right = pivot_idx -1
            else:
                left = pivot_idx +1
        
        return False
    
ob1= Solution()
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target =3
ob1.searchMatrix(matrix, target)




## Largest Number
## Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.
## Since the result may be very large, so you need to return a string instead of an integer.
## Input: nums = [3,30,34,5,9]
## Output: "9534330"

## 非常巧妙，用__lt__（x,y） 重新定义或者说overload 小于

class LargeNumKey(str):
    def __lt__(x,y):
        return x+y > y+x

class Solution():
    def largestNumber(self, nums):
        nums_str = map(str, nums)
        sorted_nums_str= sorted(nums_str, key=LargeNumKey)
        largest_num = "".join(sorted_nums_str)
        return "0" if largest_num[0] ==0 else largest_num
    

ob1=Solution()
nums = [3,30,34,5,9]
ob1.largestNumber(nums)

        
s = "1234"

dp = [0 for _ in range(len(s) + 1)]
for i in range(2,2):
    print(i)
    
    
    
    
intervals = [[0,30],[5,10],[15,20]]
begin = [0]*len(intervals)
end = [0]*len(intervals)
intervals=[[13,15],[1,13]]    
class Solution:
    def minMeetingRooms(self, intervals):
        begin = [0]*len(intervals)
        end = [0]*len(intervals)
        
        for i in range(len(intervals)):
            begin[i] = intervals[i][0]
            end[i] = intervals[i][1]
            
        begin.sort()
        end.sort()
        
        count = 0
        res = 0
        i = 0 
        j = 0
        while i<len(intervals) and j<len(intervals):
            if begin[i] < end[j]:
                count +=1
                i +=1
            else:
                count -=1
                j+=1
            res = max(res, count)
        
        return res 


strs= ["Hello","World"]
s="\n".join(strs)
s.split("\n")




class Solution:
    def findMin(self, nums):
        left, right = 0, len(nums)-1
        mid = (left+right)//2
        
        if nums[0]<=nums[mid]<nums[-1]:
            return nums[0]
        
        while left<right:
            mid = (left+right)//2
            if mid-1>=0 and mid+1<=len(nums)-1 and nums[mid+1]>nums[mid] and nums[mid-1]>nums[mid]:
                return nums[mid]
            if nums[mid] >= nums[0]:
                left = mid+1
            else:
                right = mid-1
        
        return nums[left]


nums=[3,1,2]
ob1 = Solution()
ob1.findMin(nums)




class Solution:
    def calculate(self, s: str) -> int:
        inner, outer, result, opt = 0, 0, 0, '+'
        for c in s + '+':
            if c == ' ': continue
            if c.isdigit():
                inner = 10 * inner + int(c)
                continue
            if opt == '+':
                result += outer
                outer = inner
            elif opt == '-':
                result += outer
                outer = -inner
            elif opt == '*':
                outer = outer * inner
            elif opt == '/':
                outer = int(outer / inner)
            inner, opt = 0, c
        return result + outer
    
    
    
    
s="4-5*6-1"   
ob1 = Solution()
ob1.calculate(s) 



class Solution:
    def combinationSum4(self, nums, target):
        dp = [0]*(target+1)
        dp[0] = 1
        
        for i in range(1, target+1):
            for num in nums:
                if i - num >=0:
                    dp[i] += dp[i - num]
        
        return dp[target]




nums=[1,2,3]
target=4

ob1=Solution()
ob1.combinationSum4(nums, target)


## Longest Substring Without Repeating Characters
## 思路 滑动窗口 
## 记录每个字母出现的次数，如果遇到超过1次出现的字母，那么移动左侧边界，直到该字母出现次数等于1为止

from collections import Counter
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        chars = Counter()
        left = right = 0
        res = 0
        while right < len(s):
            r = s[right]
            chars[r] += 1
            
            while chars[r] > 1:
                l = s[left]
                chars[l] -= 1
                left += 1
        
            res = max(res, right - left + 1)
        
            right +=1
        
        return res 

s= "abcdecd"
ob1 = Solution()
ob1.lengthOfLongestSubstring(s)


##  Longest Palindromic Substring
## DP 二维 


class Solution:
    def longestPalindrome(self, s: str) -> str:
        dp = [ [False]*len(s) for _ in range( len(s)) ]
        
        for i in range(len(s)):
            dp[i][i]= True
            
        ans = s[0]
        
        for j in range(len(s)):
            for i in range(j):
                if s[i] == s[j] and ( dp[i+1][j-1] or j-i ==1  ):
                    dp[i][j] = True
                    if j-i+1 > len(ans):
                        ans = s[i:j+1]
        return ans


s= "abacd"
ob1 = Solution()
ob1.longestPalindrome(s)



## Container With Most Water
## 双指针， 短挡板向内推进



class Solution:
    def maxArea(self, height):
        maxarea = 0
        left = 0
        right = len(height) - 1
        
        while left < right:
            width = right - left
            true_height = min(height[left], height[right])
            maxarea = max( maxarea, width*true_height )
            
            if height[left] < height[right]:
                left +=1 
            else:
                right -=1
        
        return maxarea


height = [1,8,6,2,5,4,8,3,7]
ob1 = Solution()
ob1.maxArea(height)



## Wayfair 

A = [1, 1000, 80, -91]

def solution(A):
    res = 0
    for num in A:
        if 9 < num < 100 or -100 < num < -9:
            res += num
    return res

A = [47, 1900, 1, 90, 45]
solution(A)
            
    


import pandas as pd
import numpy as np
y_true = ["A", "B", "C", "A", "A", "B", "A", "C", "A", "A", "B", "C", "C"]
y_pred = ["A", "B", "C", "A", "B", "C", "B", "C", "A", "A", "B", "C", "C"]
weights = {"A": 0.7, "B": 0.2, "C":0.1}


from sklearn.metrics import confusion_matrix

def weighted_f1(y_true, y_pred, weights):
    precision = {}
    recall = {}
    F1 = {}
    w_f1 = 0
    res = {}
    unique_set = np.unique(y_true)
    for label in unique_set:
        y_true_new = []
        y_pred_new = []
        for item in y_true:
            if item != label:
                y_true_new.append("Others")
            else:
                y_true_new.append(label)
                
        for item in y_pred:
            if item != label:
                y_pred_new.append("Others")
            else:
                y_pred_new.append(label)
        
        cm = confusion_matrix(y_true_new, y_pred_new)
        
        pr = cm[0,0]/sum(cm[:,0])
        rc = cm[0,0]/sum(cm[0,:])
        f1 = 2*pr*rc/(pr+rc)
        w_f1 += f1* weights[label]
        
        precision[label] = round(pr,3)
        recall[label] = round(rc,3)
        F1[label] = round(f1,3)
    
    res["precision"] = precision
    res["recall"] = recall
    res["F1"] = F1
    res["weighted_F1"] = round(w_f1,3)
    
    return res

weighted_f1(y_true, y_pred, weights)



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2000, n_features=100,
                           n_informative= 20, n_redundant=0,
                           random_state=2, shuffle=True)


X_train = X[:1500]
y_train = y[:1500]
X_test = X[1500:]
y_test = y[1500:]





train_data = pd.read_csv('C:/Users/cuisa/Desktop/train.csv')
test_data =  pd.read_csv('C:/Users/cuisa/Desktop/test.csv')

train_data_x = train_data.drop(columns =["target"])
train_data_y = train_data["target"]
test_data_x = test_data.drop(columns =["target"])
test_data_y = test_data["target"]



rf_model = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2,
                             max_features = "sqrt", random_state=0)


fit = rf_model.fit(train_data_x, train_data_y)
       


y_true = test_data_y
y_pred = fit.predict(test_data_x)
fp = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted']).iloc[1,0]

score_orig=fit.score(test_data_x, test_data_y)



imp_list = []
for name in list(train_data_x):
    train_data_x_new = pd.DataFrame.copy(train_data_x)
    train_data_x_new[name]=np.random.permutation(train_data_x[name])
    new_fit = rf_model.fit(train_data_x_new, train_data_y)
    score_x= new_fit.score(test_data_x, test_data_y)
    importance_score = score_orig - score_x
    imp_list.append((name, importance_score) )

n=20
most_important = sorted(imp_list, key = lambda x:x[1], reverse=True)[:10]
mp_list = []
for tp in most_important:
    mp_list.append(tp[0])    

mp_train_data =train_data[mp_list]
mp_test_data = test_data[mp_list]



mp_fit = rf_model.fit(mp_train_data, train_data_y)
fp_most_important = pd.crosstab(test_data_y, 
                                mp_fit.predict(mp_test_data)).iloc[1,0]


print(fp)
print(most_important)
print(fp_most_important)



data_train = pd.read_csv('C:/Users/cuisa/Desktop/train.csv')
data_test =  pd.read_csv('C:/Users/cuisa/Desktop/test.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree

def run_clf(data_train, data_test, n):
    res = {}

    data_train_x = data_train.drop(columns =["target"])
    data_train_y = data_train["target"]                   ## extract target values
    data_test_x =  data_test.drop(columns =["target"])
    data_test_y =  data_test["target"]                    ## extract target values


    ## train random forest model, you can change the parameters what you want
    tree_model = tree.DecisionTreeClassifier(random_state=np.random.seed(0))

    ## train the model and get false positive predictions
    total_fit =  tree_model.fit(data_train_x, data_train_y)
    fp = pd.crosstab(data_test_y, total_fit.predict(data_test_x)).iloc[0,1]

    ## get the original score (accuracy)
    score_orig= total_fit.score(data_test_x, data_test_y)


    ## do permutation for each variable to obtain score_x and then get the importance for each variable
    imp_list = []
    for name in list(data_train_x):
        data_train_x_new = pd.DataFrame.copy(data_train_x)
        np.random.seed(0)
        data_train_x_new[name] = np.random.permutation(data_train_x_new[name])
        new_fit =  tree_model.fit(data_train_x_new, data_train_y)
        score_x = new_fit.score(data_test_x, data_test_y)
        importance_score = score_orig - score_x
        imp_list.append((name, importance_score) )

    ## get first n most important variables
    most_important = sorted(imp_list, key = lambda x:x[1], reverse=True)[:n]


    ## Finally, train the new decision tree using the most important variables
    mp_list = []
    for tp in most_important:
        mp_list.append(tp[0])    

    mp_data_train_x =data_train_x[mp_list]
    mp_data_test_x = data_test_x[mp_list]
    mp_fit =  tree_model.fit(mp_data_train_x, data_train_y)


    fp_most_important = pd.crosstab(data_test_y, 
                                    mp_fit.predict(mp_data_test_x)).iloc[0,1]

    res["fp"] = fp
    res["most_important"] = most_important
    res["fp_most_important"] = fp_most_important

    return res 

run_clf(train_data, test_data, 10)



### Wayfair 1 

def trans_235(N):
    for i in range(1,N+1):
        if i % 2 == 0 and i % 3 != 0 and i % 5 !=0:
            print("Codility")
        elif i % 2 !=0 and i % 3 == 0 and i % 5 !=0:
            print("Test")
        elif i % 2 !=0 and i % 3 !=0 and i % 5 ==0:
            print("Coders")
        elif i % 2 == 0 and i % 3 ==0 and i % 5 !=0:
            print("CodilityTest")
        elif i % 2 == 0 and i % 3 !=0 and i % 5 ==0:
            print("CodilityCoders")
        elif i % 2 != 0 and i % 3 == 0 and i % 5 ==0:
            print("TestCoders")
        elif i % 2 == 0 and i % 3 ==0 and i % 5 ==0:
            print("CodilityTestCoders")
        else:
            print(i) 
        
        
trans_235(24)        
        




## euclidean distance customized
import math

def augemented_euclidean_dist(a,b):
    valid_idx_a = set()
    valid_idx_b = set()
    for i in range(len(a)):
        if a[i] != -999:
            valid_idx_a.add(i)
        if b[i] != -999:
            valid_idx_b.add(i)
        
    common_pos = list(valid_idx_a & valid_idx_b)
    if len(common_pos) <=1:
        return math.inf
    else:
        return(math.sqrt(sum((a[common_pos]-b[common_pos])**2)))






a= np.array([1,2,3.5,4.24])
b= np.array([-999, 4, 1.2, 3])
c= np.array([2, 1, -999, -999])   
    

augemented_euclidean_dist(a, b)
augemented_euclidean_dist(a, a)
augemented_euclidean_dist(a, 2*a)
augemented_euclidean_dist(b, c)



### smallest integer greater than N, with equal sum of digits



def smallest_same_sum_digit(N):
    def digit_sum(N):
        digit_sum = 0
        while N !=0:
            remainder = N % 10
            N = N // 10
            digit_sum += remainder
        return digit_sum
    
    target = digit_sum(N)
    

    res = N+1
    while target != digit_sum(res):
        res +=1
    return res
    
        
    

smallest_same_sum_digit(28)
smallest_same_sum_digit(734)
smallest_same_sum_digit(1990)
smallest_same_sum_digit(1000)





### minimum of coins need to be reversed

def smallest_reverse(A):
    return min(A.count(1), A.count(0))

A = [1,0,0,0,1,0]
smallest_reverse(A)



### give two integers A and B, 
### returns the number of integers from the range [A..B] 
### ends are included which can be expressed as the product of two consecutive integers, that is X*(X+1)


def pronic_interval(A,B):
    N_A = math.floor(math.sqrt(A))
    N_B = math.floor(math.sqrt(B))
    if N_A*(N_A+1) < A:
        N_A += 1
    if N_B*(N_B+1) > B:
        N_B -= 1
    return N_B-N_A+1

pronic_interval(21, 29)   
pronic_interval(6, 20)      


## skip power 2 number

def skip_power2(N):
    for i in range(1, N+1):
        if i & i-1 == 0:
            print("POWER")
        else:
            print(i)

skip_power2(16)
