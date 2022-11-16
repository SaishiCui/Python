class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



## Two sums

class Solution:
    def twoSum(self, nums, target):
        ht = {}
        for i in range(len(nums)):
            if target - nums[i] not in ht:
                ht[nums[i]] = i
            else:
                return [ht[target - nums[i] ], i]
nums = [2,7,11,15]
target = 9
ob1 = Solution()
ob1.twoSum(nums, target)
        







# Meeting Rooms
# 需要注意先sort， key=lambda x:x[0]


class Solution:
    def canAttendMeetings(self, intervals):
        intervals = sorted(intervals, key=lambda x: x[0])
        if len(intervals) <=1:
            return True
        i=1
        while i <= len(intervals)-1:
            if intervals[i][0]<intervals[i-1][1]:
                return False
            i +=1
        return True 
    

## Roman to Integer
## 注意， 定义一个字典，写循环，用while比较好，因为可以控制步长，假如前一位小于后一位，那么先减再加
## 否则直接加

class Solution:
    def romanToInt(self, s):
        dic = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        i = 0
        total = 0
        while i<len(s):
            if i+1< len(s) and dic[s[i]]<dic[s[i+1]]:
                total += dic[s[i+1]] - dic[s[i]]
                i +=2
            else:
                total +=dic[s[i]]
                i +=1
                
        return total 

s = "MCMXCIV"
s = "LVIII"
s = "III"
ob1= Solution()
ob1.romanToInt(s) 


# Backspace String Compare
## 注意，这个题用stack，先进先出


class Solution:
    def backspaceCompare(self, s, t):
        
        def check(string):
            out = []
            for i in range(len(string)):
                if string[i] != "#":
                    out.append(string[i])
                elif out:
                        out.pop()

            return "".join(out )
        
        
        return check(s) == check(t)
    
s = "ab#c"
t = "ad#c"    
s = "ab##"
t = "c#d#"
s = "a#c"
t = "b"
ob1= Solution()
ob1.backspaceCompare(s, t) 


##  Valid Palindrome
## 注意.join的用法 还有.isalnum() (alphabet and number)判断的函数， 还有.lower()用法


class Solution(object):
    def isPalindrome(self, s):
        s1="".join(ch for ch in s if ch.isalnum())
        s2=s1.lower()
        return s2 == s2[::-1]





s = "A man, a plan, a canal: Panama"
ob1 = Solution()
ob1.isPalindrome(s)


## Valid Anagram
## 最简单的方法就是排序，但是时间是nlogn
## 快速的办法是建立一个count list ascii码


class Solution:
    def isAnagram(self, s, t):
        return sorted(s) == sorted(t)
    
class Solution:
    def isAnagram(self, s, t):
        count_s = [0]*26
        count_t = [0]*26
        
        for char in s:
            count_s[ord(char)-ord("a")] +=1
        for char in t:
            count_t[ord(char)-ord("a")] +=1
    
        return count_s == count_t
    

s = "anagram"
t = "nagaram"
ob1 = Solution()
ob1.isAnagram(s, t)



## Binary search
## 没啥可说的，基本，但是要注意left<=right


class Solution:
    def search(self, nums, target):
        left = 0 
        right = len(nums) - 1
        
        while left <= right:
            mid = left + (right-left)//2
            if target > nums[mid]:
                left = mid+1
            elif target < nums[mid]:
                right = mid-1
            else:
                return mid 
        return -1
        
        


nums = [-1,0,3,5,9,12]
target = 9
ob1 = Solution()
ob1.search(nums, target)








## First Bad Version
## Binary search的变形 跟上述二分法一样，只不过注意边际条件

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n
        
        while left<=right:
            mid = left+(right-left)//2
            if not isBadVersion(mid):
                left = mid +1
            elif isBadVersion(mid):
                out = mid
                right = mid-1
        
        return out 







## flood fill
## 注意dfs



class Solution:
    def floodFill(self, image, sr, sc, color):
        if image[sr][sc] == color:
            return image
        
        need_to_change_color = image[sr][sc]
        def dfs(r,c):
            if image[r][c] == need_to_change_color:
                image[r][c] = color
                if r-1>=0:
                    dfs(r-1,c)
                if r+1<len(image):
                    dfs(r+1,c)
                if c-1>=0:
                    dfs(r,c-1)
                if c+1<len(image[0]):
                    dfs(r,c+1)
        dfs(sr,sc)
        return image
    
                
                


image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1
sc = 1
color = 2

ob1 = Solution()
ob1.floodFill(image, sr, sc, color)


## Maximum Subarray
## 动态规划，但是要注意dp[n]代表什么
## dp[i] 要代表含有nums[i]的subarray
## 一共有两种情况，第一种情况是nums[i]与前面拼接
## 第二种情况是自成一派
## 注意，最后要遍历整个dp找到最大值

class Solution:
    def maxSubArray(self, nums):
        
        n = len(nums)
        dp_old = nums[0]
        maxnum= nums[0]
        
        for i in range(1,n):
            dp_new = max(dp_old+nums[i],nums[i])
            maxnum = max(maxnum, dp_new)
            dp_old = dp_new
        
        return maxnum
    
    
    
    
## Lowest Common Ancestor of a Binary Search Tree
## 注意，输入是3个treenode, 输出是一个treenode
## preorder 前序判断，如果两个点的值都大于parent节点，那么两个点一定在右子树
## 如果两个点的值都小于parent节点，那么两个点一定在左子树

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        
        parent = root.val
        p_val = p.val 
        q_val = q.val
        
        if p_val > parent and q_val > parent:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_val < parent and q_val < parent:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
        

## Balanced Binary Tree
## 思路，计算每个点的最大深度，然后做判断
## 后序postorder, 但是递归之间要加上判断，假如有一个子树不balanced，那么直接停止递归
## helper函数要返回两种值，第一个是false或者true， 第二个是树的深度



class Solution:
    def isBalancedHelper(self, root):
        if not root:
            return True, -1
        
        leftcheck, leftheight = self.isBalancedHelper(root.left)
        if not leftcheck:
            return False, "Stop"
        
        rightcheck, rightheight = self.isBalancedHelper(root.right)
        if not rightcheck:
            return False, "Stop"
        
        return (abs(leftheight-rightheight)<2 ), 1+max(leftheight, rightheight)
        
        
        
    def isBalanced(self, root):
        return self.isBalancedHelper(root)[0]


root = TreeNode(4, left=TreeNode(3, left=TreeNode(1), right=TreeNode(2)) , right = TreeNode(5))
ob1 = Solution()
ob1.isBalanced(root)


## Linked List Cycle
## 注意，node如果循环，是一样的在set里面

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        node_seen=set()
        while head is not None:
            if head in node_seen:
                return True
            else:
                node_seen.add(head)
                head=head.next
        return False 
    
node2= ListNode(2, next= ListNode(0, next= ListNode(4)))
node2= ListNode(2, next= ListNode(0, next= ListNode(4, next= node2)))

node1 = ListNode(3, next=node2 )


## Implement Queue using Stacks
## 用两个栈stack， 来实现一个队列，队列的意思就是先进的先出去，栈是先进去的最后出去
## 用stack2来存放，stack1用来中转


class MyQueue:

    def __init__(self):
        self.stack1= []
        self.stack2= []

    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop() )
            return self.stack2.pop()
        else:
            return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2[-1]
        else:
            return self.stack2[-1]
        

    def empty(self):
        return not self.stack1 and not self.stack2


## Ransom Note
## 两种方法，第一种可以用ord函数，记录26个字母出现的次数
## 第二种用stack


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        count = [0]*26
        for char in magazine:
            count[ord(char)-ord('a')]+=1
        for char in ransomNote:
            count[ord(char)-ord('a')]-=1
        for i in range(26):
            if count[i] <0:
                return False
        
        return True



class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        stack=[]
        for char1 in magazine:
            stack.append(char1)
        for char2 in ransomNote:
            if char2 in stack:
                stack.remove(char2)
            else:
                return False
        return True
    
    
    
ransomNote = "aa"
magazine = "aab"
ob1 = Solution()
ob1.canConstruct(ransomNote, magazine)


## Climbing Stairs
## 一维动态规划
## 找出递推公式
## 注意n=1的情况


class Solution:
    def climbStairs(self, n: int) -> int:
        if n ==1:
            return 1
        
        dp = [0]*n
        dp[0]=1
        dp[1]=2
        
        for i in range(2,n):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[-1]



## Longest Palindrome
## 没啥好说的，建立一个哈希表，遇到单数的减去一，并且记录下来这个一，遇到双数不用管

class Solution:
    def longestPalindrome(self, s: str) -> int:
        dic = dict()
        single =0
        for char in s:
            dic[char] = dic.get(char,0) +1
        for char in s:
            if dic[char] % 2 !=0:
                dic[char] -=1
                single = 1
        return sum(dic.values())+single
                
                
        
s = "abccccdd"
ob1 =Solution()
ob1.longestPalindrome(s)     
            


## Reverse Linked List
## 注意要设立中间变量 next_stage 和 prev

class Solution:
    def reverseList(self, head):
        curr = head
        prev = None
        while curr:
            next_stage = curr.next
            curr.next = prev
            prev = curr
            curr = next_stage
        return prev



## Majority Element
## O(N)时间，O(1)空间
## Boyer-Moore Voting Algorithm
## 正负电荷法
## majority element相当于正电荷，其他的相当于负电荷

class Solution:
    def majorityElement(self, nums):
        count = 0
        
        for num in nums:
            if count == 0:
                candidate = num
            if candidate == num:
                count +=1
            else:
                count -=1
        return candidate 
    





nums = [2,2,1,1,1,2,2]
ob1 = Solution()
ob1.majorityElement(nums)


## Middle of the Linked List
## 快慢双指针，慢的走1步，那么快的就要走两步


class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
        
        return slow 


##  Meeting Rooms
## Given an array of meeting time intervals where 
## intervals[i] = [starti, endi], determine if a person could attend all meetings.
## 按照第一个值sort是一个好办法

class Solution:
    def canAttendMeetings(self, intervals):
        intervals = sorted(intervals, key=lambda x: x[0])
        if len(intervals) <=1:
            return True
        i=1
        while i <= len(intervals)-1:
            if intervals[i][0]<intervals[i-1][1]:
                return False
            i +=1
        return True 



intervals = [[0,30],[5,10],[15,20]]
intervals = [[7,10],[2,4]]



## Roman to Integer
## while 一般还挺好用的


class Solution:
    def romanToInt(self, s):
        dic = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        i = 0
        total = 0
        while i<len(s):
            if i+1< len(s) and dic[s[i]]<dic[s[i+1]]:
                total += dic[s[i+1]] - dic[s[i]]
                i +=2
            else:
                total +=dic[s[i]]
                i +=1
                
        return total 
    
## Counting Bits
## Given an integer n, return an array ans of length n + 1 such that 
## for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
## bitwise 运算中“和”的用法，“和”的意思是，除非两个都是1，否则0.
## 一般情况下，用&来计算几个1问题


class Solution:
    def countBits(self, n):
        out = [0]*(n+1)
        for i in range(1,n+1):
            out[i] = out[i&(i-1)] + 1
        return out 

ob1 = Solution()
n=10
ob1.countBits(n)
        


##  Same Tree
## 前序遍历判断， 左等于左，右等于右


class Solution:
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False 
        
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
         

strs = ["flower","flow","flight"]
shortest = min(strs, key=len)



## Longest Common Prefix
## Write a function to find the longest common prefix string amongst an array of strings.
## If there is no common prefix, return an empty string "".

## Input: strs = ["flower","flow","flight"]
## Output: "fl"
## 注意，先把长度最小的单词找出来，然后对这个单词进行 for loop

class Solution:
    def longestCommonPrefix(self, strs):
        
        if not strs:
            return ''
        
        short_word = min(strs, key = len)
        
        for i, char in enumerate(short_word):
            for word in strs:
                if word[i] != char:
                    return short_word[:i]
        
        return short_word
   
    
## Single Number
## Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
## Input: nums = [4,1,2,1,2]
## Output: 4
    
    
    
class Solution:
    def singleNumber(self, nums):
        ht = {}
        
        for i in range(len(nums)):
            ht[nums[i]] = ht.get(nums[i],0)+1
        for key in ht.keys():
            if ht[key] == 1:
                return key

            


nums = [4,1,2,1,2]
ob1 = Solution()
ob1.singleNumber(nums)


class Solution:
    def singleNumber(self, nums):
        a=0
        for i in range(len(nums)):
            a = a^nums[i]
        
        return a 
    
    

## Palindrome Linked List
## 快慢指针
## 翻转
## 比对

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow,fast=head,head
        while fast!=None and fast.next!=None:
            slow=slow.next
            fast=fast.next.next

        pre,cur=None,slow
        while cur!=None:
            nxt= cur.next
            cur.next=pre
            pre=cur
            cur=nxt

        while pre!=None:
            if pre.val!=head.val:
                return False
            pre=pre.next
            head=head.next
        return True
        
        
head = ListNode(1, next=ListNode(2,  next=ListNode(3, next= ListNode(2, next=ListNode(1))) ) )
ob1 =Solution()
ob1.isPalindrome(head)


## Move Zeroes
## Given an integer array nums, 
## move all 0's to the end of it while maintaining the relative order of the non-zero elements.
## Note that you must do this in-place without making a copy of the array.
## Input: nums = [0,1,0,3,12]
## Output: [1,3,12,0,0]


## 快慢指针， 快指针遇到0跳位， 如果没遇到0，那么赋值给慢指针

class Solution:
    def moveZeroes(self, nums):
        slow, fast = 0, 0
        while fast <= len(nums)-1:
            if nums[fast] == 0:
                fast +=1
            else:
                nums[slow] = nums[fast]
                slow +=1
                fast +=1
                
        for i in range(slow, len(nums)):
            nums[i] = 0
            

ob1 = Solution()
nums = [0,1,0,3,12]
ob1.moveZeroes(nums)
print(nums)




##  Symmetric Tree
## Given the root of a binary tree, 
## check whether it is a mirror of itself (i.e., symmetric around its center).

## preorder 前序遍历判断
## helper() 函数
## return 左树的左等于右树的右，并且左树的右等于右树的左

class Solution:
    def isSymmetric(self, root):
        
        def helper(left, right):
            
            if not left or not right:
                return left == right
            
            if left.val != right.val:
                return False
            
            return helper(left.left, right.right) and helper(left.right, right.left)
        
        if not root:
            return True
        
        return helper(root.left, root.right)



## Missing Numbe
## Given an array nums containing n distinct numbers in the range [0, n], 
## return the only number in the range that is missing from the array.
## bitwise 位数运算^ 同样的数字异或运算，等于0， 一个数字和0异或运算，是它本身

class Solution:
    def missingNumber(self, nums):
        
        n = len(nums)
        res = n
        
        for i in range(len(nums)):
            res ^= i ^ nums[i]
            
        return res
    
    
## Convert Sorted Array to Binary Search Tree
## Input: nums = [-10,-3,0,5,9]
## Output: [0,-3,9,-10,null,5]
## 看到sorted， 一般用二分法
    
class Solution:
    def sortedArrayToBST(self, nums):
        if not nums:
            return
        
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        
        
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        
        return root 


## Find Pivot Index
## 思路： 先计算总sum， 只要知道左半边的sum，有半部分就可以用减法得到，这样只需要O(n)时间

class Solution:
    def pivotIndex(self, nums):
        S = sum(nums)
        lefttotal = 0 
        for i in range(len(nums)):
            if lefttotal == (S - lefttotal - nums[i]):
                return i
            lefttotal += nums[i]
        return -1


nums = [1,7,3,6,5,6]
nums = [1,2,3]
nums = [2, 1, -1]
ob1 = Solution()
ob1.pivotIndex(nums)



## Running Sum of 1d Array
## 思路： 无

class Solution:
    def runningSum(self, nums):
            for i in range(1,len(nums)):
                nums[i] = nums[i-1]+nums[i]
            return nums


s="A BIG BOY"
s= s.lower()
for char in s:
    print(char)
    
    
    
class Solution(object):
    def isPalindrome(self, s):
        s= s.lower()
        s_del = ""
        for char in s:
            if char.isalnum():
                s_del += char 
        return s_del == s_del[::-1] 
    
s = "A man, a plan, a canal: Panama"
ob1 = Solution()
ob1.isPalindrome(s)



## Isomorphic Strings
## 思路: 把string化为数字类型的string，并且用空格占位


class Solution:
    
    def transformString(self, s):
        index_mapping = {}
        new_str = []
        
        for i, c in enumerate(s):
            if c not in index_mapping:
                index_mapping[c] = i
            new_str.append(str(index_mapping[c]) )
        
        return " ".join(new_str)
    
    def isIsomorphic(self, s, t):
        return self.transformString(s) == self.transformString(t)
    

ob1 = Solution()
s = "egg"
t= "add"
ob1.isIsomorphic(s, t)
    



## Is Subsequence
## 思路： 双指针



class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        left_boundary = len(s)
        right_boundary = len(t)
        p_left = 0
        p_right = 0
        
        while p_left < left_boundary and p_right < right_boundary:
            if s[p_left] == t[p_right]:
                p_left += 1
            p_right +=1
        
        return p_left  == left_boundary
    
ob1 = Solution()
s = "abc"
t = "ahbgdc"
ob1.isSubsequence(s, t)



## Merge Two Sorted Lists

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
        
class Solution:
    def mergeTwoLists(self, list1, list2):
        if list1 is None:
            return list2
        if list2 is None:
            return list1
        
        if list1.val < list2.val:
            temp = head = ListNode(list1.val)
            list1 = list1.next
        else:
            temp = head = ListNode(list2.val)
            list2 = list2.next
            
        while list1 is not None and list2 is not None:
            if list1.val < list2.val:
                temp.next = ListNode(list1.val)
                list1 = list1.next
            else:
                temp.next = ListNode(list2.val)
                list2 = list2.next
            temp = temp.next 
        
        while list1 is not None:
            temp.next = ListNode(list1.val)
            list1 = list1.next
            temp = temp.next 
        while list2 is not None:
            temp.next = ListNode(list2.val)
            list2 = list2.next
            temp = temp.next 
        return head 
            

