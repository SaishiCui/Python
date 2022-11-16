
# 1. Combination Sum (backtrack)

class Solution:
    def combinationSum(self, candidates, target):

        results = []

        def backtrack(remain, comb, start):
            if remain == 0:
     
                results.append(list(comb))
                return
            
            elif remain < 0:

                return

            for i in range(start, len(candidates)):
    
                comb.append(candidates[i])
    
                backtrack(remain - candidates[i], comb, i)

                comb.pop()

        backtrack(target, [], 0)

        return results
    
    
    
candidates = [2,3,6,7]
target = 7  
ob1=Solution()
ob1.combinationSum(candidates, target)



# 2. Permutation (backtrack)

class Solution:
    def permute(self, nums):

        def backtrack(first = 0):
            # if all integers are used up
            if first == n:  
                output.append(nums[:])
            for i in range(first, n):
                # place i-th integer first 
                # in the current permutation
                nums[first], nums[i] = nums[i], nums[first]
                # use next integers to complete the permutations
                backtrack(first + 1)
                # backtrack
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        output = []
        backtrack()
        return output


nums = [1,5,11,5]
ob1=Solution()
ob1.permute(nums)





# 3. Merge Intervals

class Solution:
    def merge(self, intervals):

        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
        
        
        
intervals = [[1,3],[2,6],[8,10],[15,18]]   
intervals = [[1,3],[0,6],[0,3],[11,12],[13,14], [15,18],[17,22],[23,25]]
intervals = [[1,4],[4,5]]
intervals = [[1,3]]
intervals = [[1,4],[5,6]]
intervals = [[1,4],[0,6]]

ob1=Solution()
ob1.merge(intervals)



# 4.  Lowest Common Ancestor of a Binary Tree

class Solution:

    def __init__(self):
        # Variable to store LCA node.
        self.ans = None

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def recurse_tree(current_node):

            # If reached the end of a branch, return False.
            if not current_node:
                return False

            # Left Recursion
            left = recurse_tree(current_node.left)

            # Right Recursion
            right = recurse_tree(current_node.right)

            # If the current node is one of p or q
            mid = current_node == p or current_node == q

            # If any two of the three flags left, right or mid become True.
            if mid + left + right >= 2:
                self.ans = current_node

            # Return True if either of the three bool values is True.
            return mid or left or right

        # Traverse the tree
        recurse_tree(root)
        return self.ans



# 5. Sort Colors

class Solution(object):
    def sortColors(self, nums):
        list_0 =[]
        list_1 =[]
        list_2 =[]
        
        for i in range(len(nums)):
            if nums[i] == 0:
                list_0.append(0)
            elif nums[i] ==1:
                list_1.append(1)
            else:
                list_2.append(2)
        total_list = list_0+list_1+list_2
        
        return total_list
    
nums = [2,0,2,1,1,0]
ob1=Solution()
ob1.sortColors(nums)        


class Solution:
    def sortColors(self, nums):

        p0 = curr = 0

        p2 = len(nums) - 1

        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[p2] = nums[p2], nums[curr]
                p2 -= 1
            else:
                curr += 1
                
    
        
# 6. Word Break (dynamic programming )

class Solution(object):
    def wordBreak(self, s, wordDict):
        wordset = set(wordDict)
        dp = [False]*(len(s)+1)
        dp[0] = True
        
        for i in range(1, len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordset:
                    return True
                    break
        return dp[len(s)]
        
s = "applepenapple"
wordDict = ["apple","pen"]
s = "leetcode"
wordDict = ["leet","code"]
s = "catsandog"
wordDict = ["cats","dog","sand","and","cat"]
ob1=Solution()
ob1.wordBreak(s, wordDict)




        

