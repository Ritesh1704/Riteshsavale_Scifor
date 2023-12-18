#!/usr/bin/env python
# coding: utf-8

# # test
# # RITESH S SAVALE

# Question: Reverse a string using a for loop in Python.

# In[11]:


string=input("Enter a string")
reversed_string = ""

for i in range(len(string)-1, -1, -1):
    reversed_string += string[i]

print("reverse string",reversed_string)


# In[5]:


list=[1,2,3,4,5,6,7,8,9,10]
sum=0
for i in range(len(list)):
    sum=sum+list[i]
print("total sum of the numbers:",sum)

    


# - Question: Write a Python program that checks whether a given number is even or odd using an if-else statement

# In[15]:


number=int(input("Enter an integer"))
if(number%2==0):
    print(number,"is an even number.")
else:
     print(number,"is an odd number.")


# - Question: Implement a program to determine if a year is a leap year or not using if-elif-else statements.

# In[2]:


year=int(input("Enter the year"))
if(year%4==0):
    print(year,"is a leap year.")
else:
     print(year,"is not a leap year.")


# - Question: Use a lambda function to square each element in a list

# In[10]:


original_list = [1, 2, 3, 4, 5]

squared_list = list(map(lambda x: x**2, original_list))

print(squared_list)


# - Question: Write a lambda function to calculate the product of two

# In[2]:


a=int(input("enter first number:"))
b=int(input("enter second number:"))
x=lambda a,b:a*b
print("Multiplication of two number is :",x(a,b))

