## 1. Letter Combinations of a Phone Number


class Solution(object):
    def letterCombinations(self, digits):
        
        dic = {"2":["a", "b", "c"], "3":["d", "e", "f"], "4":["g", "h", "i"],
               "5":["j", "k", "l"], "6":["m", "n", "o"], "7":["p","q","r", "s"],
               "8":["t","u","v"], "9":["w", "x", "y", "z"]}
        digitlist=[]
        for digit in digits:
            digitlist.append(digit)
        
        def combine(A,B):
            outlist=[]
            for charA in A:
                for charB in B:
                    outlist.append(charA+charB)
            return outlist
        
        if len(digits)==1:
            return dic[digits]
        if len(digits)==0:
            return []
        if len(digits)>1:
            while len(digitlist)>1:
                A = dic[digitlist[0]]
                B = dic[digitlist[1]]
                digitlist.insert(0, digitlist[0]+digitlist[1])
                digitlist.pop(1)
                digitlist.pop(1)
                dic[digitlist[0]]= combine(A, B)
        return dic[digitlist[0]]





digits = "2345"

ob1 = Solution()
ob1.letterCombinations("238")
        


   ## (second solution backtrack)

class Solution(object):
    def letterCombinations(self, digits):
        
        if len(digits) == 0:
            return []
        
        letters = {"2":["a", "b", "c"], "3":["d", "e", "f"], "4":["g", "h", "i"],
               "5":["j", "k", "l"], "6":["m", "n", "o"], "7":["p","q","r", "s"],
               "8":["t","u","v"], "9":["w", "x", "y", "z"]}

        def backtrack(index, path):
            if len(path) == len(digits):
                combinations.append("".join(path))
                return
            
            possible_letters = letters[digits[index]]
            
            for letter in possible_letters:
                path.append(letter)
                
                backtrack(index+1, path)
                
                path.pop()
                
        combinations=[]
        backtrack(0, [])
        return combinations    



# 2. Word Search  (backtrack)

class Solution(object):
    def exist(self, board, word):
        self.rows = len(board)
        self.cols = len(board[0])
        self.board = board
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.backtrack(row, col, word):
                    return True
        return False
        
    def backtrack(self, row, col, suffix):
        if len(suffix) == 0:
            return True 
        if row<0 or row == self.rows or col<0 or col == self.cols \
            or self.board[row][col] != suffix[0]:
                return False 
        ret = False
        self.board[row][col] = "#"
        
        for rowoffset, coloffset in [(0,1), (1,0), (0,-1), (-1,0)]:
            ret = self.backtrack(row+rowoffset, col+coloffset, suffix[1:] )
            
            if ret:
                break
        self.board[row][col]=suffix[0]
        
        return ret
    
        
                    
            

        

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCB"

board=[["a","b"],["c","d"]]
word="cdba"


board = [["A","B","C","E"],["S","F","E","S"],["A","D","E","E"]]
word="ABCESEEEFS"

ob1= Solution()
ob1.exist(board, word) 




# 3.Find All Anagrams in a String
from collections import Counter
class Solution(object):
    def findAnagrams(self, s, p):
        
        ns = len(s)
        np = len(p)
        
        if np>ns:
            return []
        out = []
        p_count = Counter(p)
        s_count = Counter()
        
        for i in range(ns):
            s_count[s[i]] += 1
            
            if i>=np:
                if s_count[s[i-np] ]==1:
                    del s_count[s[i-np] ]
                else:
                    s_count[ s[i-np]] -=1
            
            if p_count == s_count:
                out.append(i-np+1)
            
        return out

s = "cbaebabacd"
p = "abc"     
s="abab"   
p= "ab"
ob1 = Solution()
ob1.findAnagrams(s, p)

# 4. Minimum Height Trees

class Solution:
    def findMinHeightTrees(self, n, edges):

        # edge cases
        if n <= 2:
            return [i for i in range(n)]

        # Build the graph with the adjacency list
        neighbors = [set() for i in range(n)]
        for start, end in edges:
            neighbors[start].add(end)
            neighbors[end].add(start)

        # Initialize the first layer of leaves
        leaves = []
        for i in range(n):
            if len(neighbors[i]) == 1:
                leaves.append(i)

        # Trim the leaves until reaching the centroids
        remaining_nodes = n
        while remaining_nodes > 2:
            remaining_nodes -= len(leaves)
            new_leaves = []
            # remove the current leaves along with the edges
            while leaves:
                leaf = leaves.pop()
                # the only neighbor left for the leaf node
                neighbor = neighbors[leaf].pop()
                # remove the only edge left
                neighbors[neighbor].remove(leaf)
                if len(neighbors[neighbor]) == 1:
                    new_leaves.append(neighbor)

            # prepare for the next round
            leaves = new_leaves

        # The remaining nodes are the centroids of the graph
        return leaves
    
n = 6
edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]

 ##  bfs but runtime is long 
import copy
class Solution:
    def findMinHeightTrees(self, n, edges):
        if n <= 2:
            return [i for i in range(n)]
        
        neighbors = [set() for i in range(n)]
        for start, end in edges:
            neighbors[start].add(end)
            neighbors[end].add(start)
            
        def bfs(root, neighbors):   
            neighbors2 = copy.deepcopy(neighbors)
            neighbors3 = copy.deepcopy(neighbors)
            height = 0
            while neighbors2 != [set() for i in range(n)]:
                for node in neighbors2[root]:
                    neighbors3[root].remove(node)
                    neighbors3[node].remove(root)
                    if len(neighbors3[node])>=1:
                        root2 = node
                if neighbors3 == [set() for i in range(n)]:
                    break    
                root = root2
                height +=1
                neighbors2=copy.deepcopy(neighbors3) 
                
            return height
        out = []
        
        for i in range(n):
            out.append(bfs(i, neighbors))
        
        mini = min(out)
        outnumber = []
        for i in range(n):
            if out[i]==mini:
                outnumber.append(i)
        
        return outnumber
    
n = 6
edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
edges = [[1,0],[1,2],[1,3]]
n=4
ob1=Solution()
ob1.findMinHeightTrees(n, edges)


# 5.  Task Scheduler

class Solution:
    def leastInterval(self, tasks, n):
        # frequencies of the tasks
        frequencies = [0] * 26
        for t in tasks:
            frequencies[ord(t) - ord('A')] += 1
        
        frequencies.sort()

        # max frequency
        f_max = frequencies.pop()
        idle_time = (f_max - 1) * n
        
        while frequencies and idle_time > 0:
            idle_time -= min(f_max - 1, frequencies.pop())
        idle_time = max(0, idle_time)

        return idle_time + len(tasks)
    
    
tasks = ["A", "B", "A", "A", "B", "C", "A","A"]
tasks = ["A", "A", "A", "B", "B", "B"]
n=1




### permutation follow-up unique 
class Solution(object):
    def permuteUnique(self, nums):
        def backtrack(first = 0):
            if first == n and nums[:] not in output:  
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




class Solution:
    def permuteUnique(self, nums):
        results = []
        def backtrack(comb, counter):
            if len(comb) == len(nums):
                # make a deep copy of the resulting permutation,
                # since the permutation would be backtracked later.
                results.append(list(comb))
                return

            for num in counter:
                if counter[num] > 0:
                    # add this number into the current combination
                    comb.append(num)
                    counter[num] -= 1
                    # continue the exploration
                    backtrack(comb, counter)
                    # revert the choice for the next exploration
                    comb.pop()
                    counter[num] += 1

        backtrack([], Counter(nums))

        return results


nums = [1,2,3,4]
ob1 =Solution()
ob1.permuteUnique(nums)



# 6.  Minimum Window Substring

def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """

    if not t or not s:
        return ""

    # Dictionary which keeps a count of all the unique characters in t.
    dict_t = Counter(t)

    # Number of unique characters in t, which need to be present in the desired window.
    required = len(dict_t)

    # left and right pointer
    l, r = 0, 0

    # formed is used to keep track of how many unique characters in t are present in the current window in its desired frequency.
    # e.g. if t is "AABC" then the window must have two A's, one B and one C. Thus formed would be = 3 when all these conditions are met.
    formed = 0

    # Dictionary which keeps a count of all the unique characters in the current window.
    window_counts = {}

    # ans tuple of the form (window length, left, right)
    ans = float("inf"), None, None

    while r < len(s):

        # Add one character from the right to the window
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1

        # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1

        # Try and contract the window till the point where it ceases to be 'desirable'.
        while l <= r and formed == required:
            character = s[l]

            # Save the smallest window until now.
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            # The character at the position pointed by the `left` pointer is no longer a part of the window.
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1

            # Move the left pointer ahead, this would help to look for a new window.
            l += 1    

        # Keep expanding the window once we are done contracting.
        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

s = "ADOBECODEBANC"
t = "ABC"



# 7. Kth Smallest Element in a BST

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
        
class Solution(object):
    def kthSmallest(self, root, k):
        def inorder(root):
            return inorder(root.left) + [root.val] + inorder(root.right) if root else []
    
        return inorder(root)[k+1]
    
root = TreeNode(5, left = TreeNode(3, left=TreeNode(2), right=TreeNode(4)   ) ,
                right = TreeNode(6, right=TreeNode(7) ))

