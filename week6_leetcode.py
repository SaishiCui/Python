# 1. Partition Equal Subset Sum (dynamic programming)

 
class Solution:
    def canPartition(self, nums):
        # find sum of array elements
        total_sum = sum(nums)

        # if total_sum is odd, it cannot be partitioned into equal sum subsets
        if total_sum % 2 != 0:
            return False
        subset_sum = total_sum // 2

        # construct a dp table of size (subset_sum + 1)
        dp = [False] * (subset_sum + 1)
        dp[0] = True
        for curr in nums:
            for j in range(subset_sum, curr - 1, -1):
                dp[j] = dp[j] or dp[j - curr]

        return dp[subset_sum]
        
        
        
nums=[1,5,11,5]
nums = [1,2,3,5]  
ob1=Solution()
ob1.canPartition(nums)      




# 2.  String to Integer (atoi)
class Solution(object):
    def myAtoi(self, input):
        sign =1
        result = 0
        index =0
        n = len(input)
        
        INT_MAX =pow(2,31)-1
        INT_MIN = -pow(2,31)
        
        while index<n and input[index]==" ":
            index +=1
            
        if index< n and input[index] =="+":
            sign = 1
            index +=1
        elif index<n and input[index]=="-":
            sign = -1
            index +=1
            
        while index<n and input[index].isdigit():
            digit = int(input[index])
            
            if ((result>INT_MAX//10) or (result == INT_MAX//10 and digit > INT_MAX %10)):
                return INT_MAX if sign ==1 else INT_MIN
            
            result = 10*result + digit
            index +=1
            
        return sign*result
    


# 3.Spiral Matrix

class Solution(object):
    def spiralOrder(self, matrix):
        R, Rtotal = len(matrix), len(matrix)
        C, Ctotal = len(matrix[0]), len(matrix[0])
        left_bound =0
        top_bound =1
        r = 0
        c= 0
        output = []
        len1=0
        len2=1
        while len1 != len2:
            len1 =  len(output)
            while c+1 <= C:
                output.append(matrix[r][c])
                c +=1
            c -=1
            r +=1
            if Rtotal*Ctotal == len(output):
                break
            
            while r+1 <=R:
                output.append(matrix[r][c])
                r +=1
            r -=1
            c -=1
            if Rtotal*Ctotal == len(output):
                break
            
            while c>=left_bound:
                output.append(matrix[r][c])
                c -=1
            c +=1
            r -=1
            if Rtotal*Ctotal == len(output):
                break
            
            while r>=top_bound:
                output.append(matrix[r][c])
                r -=1
            if Rtotal*Ctotal == len(output):
                break
            len2 =  len(output)
            r +=1
            c +=1
            C -=1
            R -=1
            left_bound +=1
            top_bound  +=1
        return output
        
   # (solution 2)
class Solution:
    def spiralOrder(self, matrix):
        VISITED = 101
        rows, columns = len(matrix), len(matrix[0])
        # Four directions that we will move: right, down, left, up.
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Initial direction: moving right.
        current_direction = 0
        # The number of times we change the direction.
        change_direction = 0
        # Current place that we are at is (row, col).
        # row is the row index; col is the column index.
        row = col = 0
        # Store the first element and mark it as visited.
        result = [matrix[0][0]]
        matrix[0][0] = VISITED

        while change_direction < 2:

            while True:
                # Calculate the next place that we will move to.
                next_row = row + directions[current_direction][0]
                next_col = col + directions[current_direction][1]

                # Break if the next step is out of bounds.
                if not (0 <= next_row < rows and 0 <= next_col < columns):
                    break
                # Break if the next step is on a visited cell.
                if matrix[next_row][next_col] == VISITED:
                    break

                # Reset this to 0 since we did not break and change the direction.
                change_direction = 0
                # Update our current position to the next step.
                row, col = next_row, next_col
                result.append(matrix[row][col])
                matrix[row][col] = VISITED

            # Change our direction.
            current_direction = (current_direction + 1) % 4
            # Increment change_direction because we changed our direction.
            change_direction += 1

        return result           
            
        

            
matrix = [[1,2,3],[4,5,6],[7,8,9]]
matrix =[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
ob1=Solution()
ob1.spiralOrder(matrix)



# 4. Power set (backtracking similar as permutation)


class Solution:
    def subsets(self, nums):
        
        

        def backtrack(first, subset):
            # if all integers are used up
            if len(subset) == k:  
                output.append(subset[:])
                return 
            
            for i in range(first, n):
 
                subset.append(nums[i])

                backtrack(i + 1, subset)
      
                subset.pop()
        
        n = len(nums)
        output = []
        for k in range(n + 1):
            backtrack(0, [])
        return output


nums = [1,2,3]
ob1=  Solution()
ob1.subsets(nums)






# 5. Longest Palindromic Substring

import numpy as np

class Solution(object):
    def longestPalindrome(self, s):
        dp =np.array([[False]*len(s)]*len(s)  )     
        out_string = s[0]
        for i in range(len(s)):
            dp[i][i]=True
            if i+1<=len(s)-1:
                dp[i][i+1] = s[i]==s[i+1]
                if dp[i][i+1]:
                    out_string = s[i:(i+2)]

        if len(s)>2:
            for j in range(2,len(s)):
                for i in range( j-1):
                    dp[i][j]= dp[i+1][j-1] and s[i]==s[j]
                    if dp[i][j]:
                        if len(s[i:(j+1)])>len(out_string):
                            out_string = s[i:(j+1)]
        
                
        return out_string

                        
s = "vckpzcfezppubykyxvwhbwvgezvannjnnxgaqvesrhdsgngcbbdpqeodzmqbkrwekakrukwxhqjeacxhkixruwshgwkjthmtqumvqcvhhoavarlwhpzbbniqrswvyhtfquioooejsbnxdnjrfhzpdrljcuenzjpzkyrgpxrbtchnzmdkekhmuqpoljvrpndzmogeuxjotdsyrrudligpgwcblaimqdqsgmjrbvyonugzsbkdhawmewiaccuvfnpftcjdjuljekiaipknorknwyx"                      
ob1= Solution() 
ob1.longestPalindrome(s)               




# 6. Unique Paths
import numpy as np

class Solution(object):
    def uniquePaths(self, m, n):
        dp = np.array([[1]*n]*m)
        
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j]= dp[i-1][j]+dp[i][j-1]
        return dp[m-1][n-1]
        
    
ob1=Solution()
ob1.uniquePaths(3,7)
                    
        


# 7. Accounts Merge
from collections import defaultdict
class Solution(object):
    def accountsMerge(self, accounts):
        visited_accounts = [False] * len(accounts)
        emails_accounts_map = defaultdict(list)    
        res=[]
        for i, account in enumerate(accounts):
            for j in range(1, len(account)):
                email = account[j]
                emails_accounts_map[email].append(i)
                
        def dfs(i, emails):
                   if visited_accounts[i]:
                       return
                   visited_accounts[i] = True
                   for j in range(1, len(accounts[i])):
                       email = accounts[i][j]
                       emails.add(email)
                       for neighbor in emails_accounts_map[email]:
                           dfs(neighbor, emails)
                           
               # Perform DFS for accounts and add to results.
        for i, account in enumerate(accounts):
            if visited_accounts[i]:
                continue
            name, emails = account[0], set()
            dfs(i, emails)
            res.append([name] + sorted(emails))
            
        return res
            
accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
ob1=Solution()
ob1.accountsMerge(accounts)



## 8. Container With Most Water

class Solution(object):
    def maxArea(self, height):
        maxarea = 0
        left = 0
        right = len(height)-1
        
        while left<right:
            width = right - left
            maxarea = max(maxarea, min(height[left],height[right])*width )
            if height[left] <= height[right]:
                left +=1
            else:
                right -=1
        return maxarea
    
    
ob1 = Solution()
height = [1,8,6,2,5,4,8,3,7]
ob1.maxArea(height)
                
        
        
# 9.


class TreeNode(object):
     def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def buildTree(self, preorder, inorder):

        def array_to_tree(left, right):
            nonlocal preorder_index
            # if there are no elements to construct the tree
            if left > right: return None

            # select the preorder_index element as the root and increment it
            root_value = preorder[preorder_index]
            root = TreeNode(root_value)


            preorder_index += 1

            # build left and right subtree
            # excluding inorder_index_map[root_value] element because it's the root
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)

            return root

        preorder_index = 0

        # build a hashmap to store value -> its index relations
        inorder_index_map = {}
        for index, value in enumerate(inorder):
            inorder_index_map[value] = index

        return array_to_tree(0, len(preorder) - 1)

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
ob1=Solution()
a=ob1.buildTree(preorder, inorder)



a=TreeNode(4,1,2)
