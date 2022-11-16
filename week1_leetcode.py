
#1. twoSum problem
def twoSum(num_list, target):
    output_list=[]
    ht1={}
    ht2={}
    for i in range(len(num_list)):      
        if target-num_list[i] in ht1:
            if num_list[i] not in ht2 and num_list[i] not in ht1:
                output_list.append([ ht1[target-num_list[i]], num_list[i]])
                ht2[num_list[i]]=num_list[i]
            else:
                None
        else:
            ht1[num_list[i]]=num_list[i]
    return output_list


num_list=[2,8,12,18,2]
target=20

twoSum(num_list,target)






#2. Valid Parentheses  

class Solution:
    def isValid(self, s):
        if len(s) % 2 != 0:
            return False
        dict = {'(' : ')', '[' : ']', '{' : '}'}
        stack = []
        for i in s:
            if i in dict.keys():
                stack.append(i)
            else:
                if stack == []:
                    return False
                a = stack.pop()
                if i!= dict[a]:
                    return False
        return stack==[]

ob1=Solution()

            
s="([])"
ob1.isValid(s)






##3. Best Time to Buy and Sell Stock



def maxProfit(prices):
    profitlist=[] 
    for i in range(len(prices)-1):
        for j in range(i+1,len(prices)):
            profitlist.append(prices[j]-prices[i])
    largest=max(profitlist)
    if largest<=0:
        return 0
    else:
        return largest
    
prices=[7,6,4,3,1]
maxProfit(prices)


def max_profit(prices):
    if not prices:
        return 0

    max_prof = 0
    min_price = prices[0]

    for i in range(1, len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
        max_prof = max(max_prof, prices[i] - min_price)
    return max_prof

prices=[7,6,4,3,1]
max_profit(prices)



#4. Maximum Subarray    

def maxSubArray(nums):

    current_subarray = nums[0]
    max_subarray = nums[0]
        
    for num in nums[1:]:
        current_subarray = max(num, current_subarray+num)
        max_subarray = max(max_subarray, current_subarray)
    return max_subarray 







#5. Valid Palindrome

s = "A man, a plan, a canal: Panama"
def isPalindrome(s):
    s1="".join(ch for ch in s if ch.isalnum())
    s2=s1.lower()
    s3=s2[::-1]
    if s2==s3:
        return True
    else:
        return False
    
s="232e"
isPalindrome(s)
    






    
#6.  Valid Anagram

def isAnagram(s, t):
    if len(s)==len(t):
        s_list=[]
        for char in s:
            s_list.append(char)
        for char in t:
            if char in s_list:
               s_list.remove(char)
        if len(s_list)==0:
            return True
        else:
            return False
    else:
        return False
    
s = "anagram"
t = "nagaram"
s = "rat"
t = "car"
s= "ab"
t= "a"
s= "a"
t= "ab"
s= "ccac"
t= "acca"
isAnagram(s, t)        
    
    
    
    
def isAnagram(s,t):
    if sorted(s)==sorted(t):
        return True
    else:
        return False

s = "anagram"
t = "nagaram"
isAnagram(s, t)
    
    
    
###7. Binary Search
 
def search(numlist, target):
    left=0
    right=len(numlist)-1
    while left<=right:
        pivot = left+ (right-left)//2
        if target == numlist[pivot]:
            return pivot
        elif target < numlist[pivot]:
            right = pivot-1
        else:
            left = pivot+1
    return -1
    
numlist= [-2,-1,0,6,7,8,9]
search(numlist, 9)
    



### 8. flood fill

def floodfill(image, sr,sc,newColor):
    R,C=len(image), len(image[0])
    color = image[sr][sc]
    if color == newColor:
        return image
    def dfs(r,c):
        if image[r][c]==color:
            image[r][c]=newColor
            if r>=1:
                dfs(r-1,c)
            if r+1<R:
                dfs(r+1,c)
            if c>=1:
                dfs(r,c-1)
            if c+1<C:
                dfs(r,c+1)
    dfs(sr,sc)
    return image
    
    
image=[[1,1,1],[1,1,0],[1,0,1]]
sr=1
sc=1
newColor=2

floodfill(image, sr, sc, newColor)  
    

    


# 9. Invert Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        
        right = self.invertTree(root.right)
        left  = self.invertTree(root.left)
        root.left = right
        root.right = left
        return root    
    
    


## 10.  Lowest Common Ancestor of a Binary Search Tree

def lowCommonAnces(root,p,q):
    p_val = p.val
    q_val = q.val
    parent_val=root.val
    
    if p_val > parent_val and q_val > parent_val:
        return lowCommonAnces(root.right, p, q)
    elif p_val < parent_val and q_val < parent_val:
        return lowCommonAnces(root.left, p, q)
    else:
        return root
    
    
    
    
    
## 11. balanced binary tree
class Solution(object):
    def height(self, root):
        # An empty tree has height -1
        if not root:
            return -1
        return 1 + max(self.height(root.left), self.height(root.right))
    
    def isBalanced(self, root):
        # An empty tree satisfies the definition of a balanced tree
        if not root:
            return True

        # Check if subtrees have height within 1. If they do, check if the
        # subtrees are balanced
        return abs(self.height(root.left) - self.height(root.right)) < 2 \
            and self.isBalanced(root.left) \
            and self.isBalanced(root.right)
            
            

# 12. Linked List Cycle

def hasCycle(head):
    nodes_seen=set()
    while head is not None:
        if head in nodes_seen:
            return True
        else:
            nodes_seen.add(head)
            head = head.next
    return False



                    




    
    
#### two sum advanced problem ###

numlist=[]
with open("C:/Users/cuisa/Desktop/python/2sum.txt") as f:
    data = f.readlines()
    for line in data:
        numlist.append(int(line.strip("\n")) )
f.close()


target=-9967
target=92
target=94
tar_list=[]
for target in range(-10000,10001):
    ht={}
    for x in numlist:
        if target - x in ht:
           if target-x != x:
               if target not in tar_list:
                   tar_list.append(target)
               else:
                   break
           else:
               None
        else:
            ht[x]=x
    print(target)












# Merge Two Sorted Lists
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        newlist=[]
        length=len(list1)+len(list2)
        while len(newlist)!=length:
            if len(list1) != 0 and len(list2) != 0:
                if list1[0]<=list2[0]:
                    newlist.append(list1[0])
                    list1=list1[1:]
                else:
                    newlist.append(list2[0])
                    list2=list2[1:]
            elif len(list1)==0:
                newlist.append(list2[0])
                list2=list2[1:]
            elif len(list2)==0:
                newlist.append(list1[0])
                list1=list1[1:]
        return newlist
    
ob1=Solution()
list1=[0,3,4,5,5,6,7,8]
list2=[0,1000]
ob1.mergeTwoLists(list1, list2)



class ListNode:
    def __init__(self, val=0, nextNode=None):
        self.val = val
        self.next = nextNode


def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    # Check if either of the lists is null
    if l1 is None:
        return l2
    if l2 is None:
        return l1
    # Choose head which is smaller of the two lists
    if l1.val < l2.val:
        temp = head = ListNode(l1.val)
        l1 = l1.next
    else:
        temp = head = ListNode(l2.val)
        l2 = l2.next
    # Loop until any of the list becomes null
    while l1 is not  None and l2 is not None:
        if l1.val < l2.val:
            temp.next = ListNode(l1.val)
            l1 = l1.next
        else:
            temp.next = ListNode(l2.val)
            l2 = l2.next
        temp = temp.next
    # Add all the nodes in l1, if remaining
    while l1 is not None:
        temp.next = ListNode(l1.val)
        l1 = l1.next
        temp = temp.next
    # Add all the nodes in l2, if remaining
    while l2 is not None:
        temp.next = ListNode(l2.val)
        l2 = l2.next
        temp = temp.next
    return head






