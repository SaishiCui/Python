# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 17:39:13 2022

@author: cuisa
"""

# Quick sort
## 分而治之算法， 用最后一个数当做pivot，反复递归
## 要注意partition 函数中， 指针是最左边开始，每次发现小于pivot, 指针需要加1，最后该指针位置也需要和pivot互换


class Solution(object):
    def partition1(self, l, r, nums):
        pointer = l 
        
        for i in range(l,r):
            if nums[i] <= nums[r]: ## using last item as pivot
                nums[i], nums[pointer] = nums[pointer], nums[i]
                pointer +=1
        nums[r], nums[pointer] = nums[pointer], nums[r]
        return pointer 
    
    ## using last item as pivot
    def Quicksort1(self, l, r, nums):
        if l<r:
            pivot = self.partition1(l, r, nums)
            self.Quicksort1(l, pivot-1, nums)
            self.Quicksort1(pivot+1, r, nums)
        return nums



    def partition2(self, l, r, nums):
        pointer = l+1
        
        for i in range(l+1,r+1):
            if nums[i] <= nums[l]: ## using first item as pivot
                nums[i], nums[pointer] = nums[pointer], nums[i]
                pointer +=1
        nums[l], nums[pointer-1] = nums[pointer-1], nums[l]
        return pointer-1
    
    ## using first item as pivot
    def Quicksort2(self, l, r, nums):
        if l<r:
            pivot = self.partition2(l, r, nums)
            self.Quicksort2(l, pivot-1, nums)
            self.Quicksort2(pivot+1, r, nums)
        return nums
    
    
    def QuickSelect(self, l, r, nums, k):
        if k>0 and k<= len(nums):
            if l<=r:
                pivot = self.partition1(l, r, nums)
                if pivot == k-1:
                    return nums[pivot]
                elif pivot > k-1:
                    return self.QuickSelect(l, pivot-1, nums,k)
                elif pivot < k-1:    
                    return self.QuickSelect(pivot+1, r, nums,k)
        else:
            return "Out of range"
    





ob1 =Solution()
nums=nums=[1,2,3,4,5,7,8,9,10,11,6]
ob1.QuickSelect(l=0, r=len(nums)-1, nums=nums, k=1)


## Quick select 
## Kth Largest Element in an Array
## Given an integer array nums and an integer k, return the kth largest element in the array.
## Note that it is the kth largest element in the sorted order, not the kth distinct element.
## You must solve it in O(n) time complexity.


import random

class Solution:
    def findKthLargest(self, nums, k):
        
        def partition(left, right, pivot_index):
            
            pivot = nums[pivot_index]
            
            # move pivot to the last place
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            
            # move smallest numbers to the left
            pointer = left
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[i], nums[pointer] = nums[pointer], nums[i]
                    pointer +=1
            
            # final movement
            
            nums[pointer], nums[right] = nums[right], nums[pointer]
            
            return pointer
        
        def select(left, right, k_smallest):
            if left == right:
                return nums[left]
            
            pivot_index = random.randint(left, right)
            
            # find the pivot position
            
            pivot_index = partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return nums[k_smallest]
            elif k_smallest < pivot_index:
                return select(left, pivot_index-1, k_smallest)
            else:
                return select(pivot_index + 1, right, k_smallest)
            
        return select(0, len(nums)-1, len(nums)- k )
    

nums = [3,2,1,5,6,4]
k = 2
ob1 = Solution()
ob1.findKthLargest(nums, k)

























## Merge sort 
## 分而治之，divide and conquer
## 先定义merge， 后定义sort
## 在sort里面使用递归， 左半部分sort一次，又半部分sort一次，然后merge两次的结果
## merge 函数里面也有讲究， b，c其中任意一个长度大于0就可以进行，如果全部大于0，需要进行判断
## 如果只有一个大于0，那么就传递该array里面的数就可以

class Solution(object):
    def merge(self, b,c):
       res_arr= []
       while len(b)>0 or len(c)>0:
             if len(b)>0 and len(c)>0:
                if b[0]<c[0]:
                   res_arr.append(b[0])
                   b=b[1:]
                else:
                   res_arr.append(c[0])
                   c=c[1:]
             elif len(b)>0:
                  res_arr.append(b[0])
                  b=b[1:]
             elif len(c)>0:
                  res_arr.append(c[0])
                  c=c[1:]
       return res_arr
    
    
    
    
    def sort(self,s):
       arr_len=len(s)
       if arr_len<=1:
          return s
       b= self.sort(s[:(arr_len//2)])
       c= self.sort(s[(arr_len//2):])
       d = self.merge(b,c)
       
       return d  
   
    
ob1 = Solution()
s= [4,3,2,1,5]
ob1.mergesort(s)




