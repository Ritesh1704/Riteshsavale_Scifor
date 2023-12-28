#!/usr/bin/env python
# coding: utf-8

# # LEET CODE QUES
# ## RITESH S SAVALE

# In[ ]:


Q1: Merge Strings Alternately


# In[1]:


class Solution(object):
    def mergeAlternately(self, word1, word2):
        merged = ""
        lenn = min(len(word1), len(word2))
        k = 0
        for i in range(0, lenn):
            merged += word1[i] + word2[i]
            k += 1

        if k < len(word1):
            for i in range(k, len(word1)):
                merged += word1[i]
        else:
            for i in range(k, len(word2)):
                merged += word2[i]

        return merged

obj = Solution()
print(obj.mergeAlternately(word1="abcg", word2="defeee"))


# - Q2: Find the Difference

# In[3]:


class Solution(object):
    def findTheDifference(self, s, t):
        for i in range(0, len(t)):
            count_t = t.count(t[i])
            count_s = s.count(t[i])
            if count_t> count_s:
                return t[i]

obj = Solution()
print(obj.findTheDifference(s="abc", t="xyz"))

     


# - 3: Valid anagram

# In[4]:


class Solution(object):
    def isAnagram(self, s, t):
        if len(t) != len(s):
            return False
        else:
            for i in range(0, len(t)):
                count_t = t.count(t[i])
                count_s = s.count(t[i])
                if count_t != count_s:
                    return False
            return True


obj = Solution()
print(obj.isAnagram(s="man", t="can"))
     


# - Repeated Substring Pattern

# In[7]:


class Solution(object):
   def repeatedSubstringPattern(self, s):
       length = len(s)

       for i in range(1, length // 2 + 1):

           if length % i == 0:
             substring = s[:i]
             repeated_string = substring * (length // i)
             if repeated_string == s:
                  return True

       return False


obj = Solution()
print(obj.repeatedSubstringPattern("abcabcabc"))  # Output: True
print(obj.repeatedSubstringPattern("abab"))       # Output: True
print(obj.repeatedSubstringPattern("aba"))        # Output: False


# -  Find the Index of the First Occurrence in a String

# In[9]:


class Solution(object):
    def strStr(self, haystack, needle):
      x=haystack.find(needle)
      return x

obj= Solution()
print(obj.strStr(haystack="banana", needle='anas'))


# - move Zeroes to end

# In[12]:


def moveZerosToEnd(l):
  for i in l:
    #look for zeros
    if int(i) == 0:
      j = nz(l,i)
      #swap zero with nonzero
      l[i], l[j] =  l[j], l[i]
  return l
    
def nz(l,i):
  #look for nonzero
  while i < len(l) and l[i] == 0:
    #progress if zero
    i += 1
  #return nonzero value
  return i
            


l = []
for i in range(5): l.append(int(input()))
moveZerosToEnd(l)
print(l)


# - Plus one

# In[13]:


class Solution(object):
    def plusOne(self, digits):
        n = len(digits)

        for i in range(n - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else:
                digits[i] = 0

        digits.insert(0, 1)
        return digits

obj = Solution()
result = obj.plusOne([9, 8,2])
print(result)


# - Sign of the Product of an Array

# In[16]:


class Solution(object):
    def arraySign(self, nums):
        count=0
        for i in nums:
            if i==0:
                return 0
            elif i<0:
                count+=1
        if count%2==0:
            return 1
        else:
            return -1


# -  Arithmetic progression

# In[15]:


class Solution(object):
    def canMakeArithmeticProgression(self, arr):
        arr.sort()
        for i in range(2, len(arr)):
            if 2 * arr[i - 1] != arr[i - 2] + arr[i]:
                return False
        return True

obj = Solution()
print(obj.canMakeArithmeticProgression([4, 5, 6]))

