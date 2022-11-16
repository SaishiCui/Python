
#1. Contains Duplicate
def containsDuplicate(nums):
    num_set=set()
    for i in range(len(nums)):
        if nums[i] in num_set:
            return True
        else:
            num_set.add(nums[i])
    return False


nums = [1,2,3,1]
containsDuplicate(nums)








# 2. Implement Queue using Stacks

class MyQueue(object):

    def __init__(self):
        self._stack1=[]
        self._stack2=[]


    def push(self, x):

        self._stack1.append(x)

        

    def pop(self):
        if self._stack2:
            return self._stack2.pop()
        else:
            while self._stack1:
                self._stack2.append(self._stack1.pop())
            return self._stack2.pop()
        
        

    def peek(self):
        if self._stack2:
            return self._stack2[-1]
        else:
            while self._stack1:
                self._stack2.append(self._stack1.pop())
            return self._stack2[-1]
        
        

    def empty(self):
        return not self._stack1 and not self._stack2
    

queue = MyQueue()
queue.push(1)
queue.push(2)
queue.push(3)
queue.pop()





## 3.First Bad Version


class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left=1
        right=n
        while left<right:
            pivot = left + (right-left)//2
            if not isBadVersion(pivot):
                left=pivot+1
            else:
                if not isBadVersion(pivot-1):
                    return pivot
                else:
                    right=pivot-1
        return left
            

    
    
    
    
    


# 4.Ransom Note
def canConstruct(ransomNote, magazine):
    stack=[]
    for char1 in magazine:
        stack.append(char1)
    for char2 in ransomNote:
        if char2 in stack:
            stack.remove(char2)
        else:
            return False
    return True

        
    
ransomNote="bg"
magazine="abcdefg"
canConstruct(ransomNote, magazine)

from collections import Counter




def canConstruct(ransomNote, magazine):
    if len(ransomNote)>len(magazine):
        return False
    letters=Counter(magazine)
    for c in ransomNote:
        if letters[c]<=0:
            return False
        else:
            letters[c] -= 1
    return True


ransomNote="bg"
magazine="abcdefg"
canConstruct(ransomNote, magazine)












# 5. climb stairs

def climstairs(n):
    if n ==1:
        return 1
    
    dp=[1]*n
    dp[0]=1
    dp[1]=2
    
    for i in range(2,n):
        dp[i]=dp[i-1]+dp[i-2]
    return dp[n-1]

climstairs(3)
climstairs(50)
    










# 6. Longest Palindrome


    
def longestPalindrome(s):
    dic = {char: s.count(char) for char in set(s)}
    count = 0 
    check_single = 0
    for val in dic.values():
        if val%2 ==1:
            check_single=1
            count += ((val-1)//2)*2
        else:
            count += (val//2)*2
    return count+check_single



s="aabbccdeef"
longestPalindrome(s)


def longestPalindrome(s):
    count = 0
    dic = {char: s.count(char) for char in set(s)}
    for val in dic.values():
        count += val // 2 * 2
        if count % 2 == 0 and val % 2 == 1:
            count += 1
    return count



s="aabbccdeef"
longestPalindrome(s)









# 7. Mini Stack

class MinStack(object):

    def __init__(self):
        self.stack = []

    def push(self, val):
        if not self.stack:
            self.stack.append((val,val))
            return 
        current_min = self.stack[-1][1]
        self.stack.append((val, min(val, current_min)))
        

    def pop(self):
        self.stack.pop()
        

    def top(self):
        return self.stack[-1][0]
        

    def getMin(self):
        return self.stack[-1][-1]


# 8. Reverse Linked List

def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_step = curr.next
        curr.next = prev
        prev = curr 
        curr = next_step
    return prev 


# 9. Majority element

def majorityElement(nums):
    for item in set(nums):
        if nums.count(item)> int(len(nums)/2):
            return item

nums = [2,2,1,1,1,2,2]
nums = [3,2,3]
majorityElement(nums)


# 10. Add Binary

def addBinary(a, b):
    return '{0:b}'.format(int(a,2)+int(b,2))


addBinary("111", "110")

def addBinary(a,b):
    return str(bin(int(a,2)+int(b,2)))[2:]
addBinary("111", "110")



# 11. diameter of binary tree

def diameterOfBinaryTree(self, root):
    diameter = 0

    def longest_path(node):
        if not node:
            return 0
        nonlocal diameter
            # recursively find the longest path in
            # both left child and right child
        left_path = longest_path(node.left)
        right_path = longest_path(node.right)

            # update the diameter if left_path plus right_path is larger
        diameter = max(diameter, left_path + right_path)

            # return the longest one between left_path and right_path;
            # remember to add 1 for the path connecting the node and its parent
        return max(left_path, right_path) + 1

    longest_path(root)
    return diameter



# 12. Middle of the Linked List
def middleNode(head):
    arr =[head]
    while arr[-1].next:
        arr.append(arr[-1].next)
    return arr[len(arr)//2]


#13. Maximum Depth of Binary Tree

def maxDepth(self, root):
    if not root:
        return 0
        
    leftdepth= self.maxDepth(root.left)
    rightdepth=self.maxDepth(root.right)
        
    return max(leftdepth, rightdepth)+1
