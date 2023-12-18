#!/usr/bin/env python
# coding: utf-8

# # functions
# # Ritesh S Savale

# - What is function?

# 
# #### A function is a named block of code that performs a specific task. It takes input (arguments), processes it, and returns an output. Functions promote code reusability and modular design.

# - Why we used function?

# 
# - `Modularity and Reusability`: Breaks code into reusable modules.
# - `Readability`: Enhances code comprehension and structure.
# - `Abstraction`: Focuses on what, not how, improving clarity.
# - `Parameterization`: Supports flexible input handling.
# - `Encapsulation`: Safely isolates functionality for easy management.
# - `Testing and Debugging`: Facilitates unit testing and targeted debugging.
# - `Namespace Isolation`: Avoids naming conflicts, enhances code safety.
# - `Code Organization`: Logical structure for scalability and maintenance.
# - `Return Values`: Produces results for data flow and computation.
# - `Lambda Functions`: Supports concise, anonymous functions.

# ### Syntax of function

# In[ ]:


'''def function_name(parameters):
    """docstring"""
    # Function body (code block)
    return result'''


# - it looks like

# ![image.png](attachment:image.png)

# ### There are two types of functions in python

# - Built-in library function: These are Standard functions in Python that are available to use.
# - User-defined function: We can create our own functions based on our requirements.

# ## Creating a Function in Python

# In[ ]:


# A simple Python function 

def fun():
print("Welcome to GFG")


# ## Calling a  Python Function

# In[1]:


# A simple Python function
def fun():
    print("Welcome to GFG")

#function call
fun()


# ## Python Function with Parameters

# - Defining and calling a function with parameters
# 

# In[ ]:


def function_name(parameter: data_type) -> return_type:
    """Docstring"""
    # body of the function
    return expression


# ## Python Function Arguments
# - Arguments are the values passed inside the parenthesis of the function. A function can have any number of arguments separated by a comma.

# In[2]:


# A simple Python function to check
# whether x is even or oddb
def evenOdd(x):
    if (x % 2 == 0):
        print("even")
    else:
        print("odd")


# Driver code to call the function
evenOdd(2)
evenOdd(3)


# ### Types of Python Function Arguments
# - Python supports various types of arguments that can be passed at the time of the function call. In Python, we have the following 4 types of function arguments.
# 
# 1.Default argument                                             
# 
# 2.Keyword arguments (named arguments)
# 
# 3.Positional arguments
# 
# 4.Arbitrary arguments (variable-length arguments *args and **kwargs)

# - Default Arguments

# In[3]:


# default arguments
def myFun(x, y=50):
    print("x: ", x)
    print("y: ", y)
 


myFun(10)


# - Keyword Arguments

# In[4]:


# Keyword Arguments
def student(firstname, lastname):
    print(firstname, lastname)
 
 
# Keyword arguments
student(firstname='Geeks', lastname='Practice')
student(lastname='Practice', firstname='Geeks')


# - Positional Arguments

# In[5]:


def nameAge(name, age):
    print("Hi, I am", name)
    print("My age is ", age)
 

print("Case-1:")
nameAge("Suraj", 27)

print("\nCase-2:")
nameAge(27, "Suraj")


# - Arbitrary arguments (variable-length arguments *args and **kwargs)

# In[6]:


# *args for variable number of arguments
def myFun(*argv):
    for arg in argv:
        print(arg)
 
 
myFun('Hello', 'Welcome', 'to', 'GeeksforGeeks')


# ## Pass by Reference and Pass by Value

# In[ ]:


# pass by value


# In[8]:


def modify_value(x):
    x = x + 10
    print("Inside function:", x)

# Call by value
num=int(input())
modify_value(num)
print("Outside function:", num)


# In[ ]:


# pass by reffrence


# In[9]:


def modify_list(lst):
    lst.append(10)
    print("Inside function:", lst)

# Pass by reference 
my_list = [1, 2, 3]
modify_list(my_list)
print("Outside function:", my_list)


# In[ ]:


# simple examples of functions


# In[12]:


#addition of two numbers
def add_numbers(x, y):
    return x + y

result = add_numbers(3, 5)
print("Sum:", result)


# In[13]:


# Square of a Number:

def square(number):
    return number ** 2

result = square(4)
print("Square:", result)


# In[14]:


# Check Even or Odd:

def is_even(number):
    return number % 2 == 0

num = 7
print(f"{num} is even: {is_even(num)}")


# In[15]:


# Factorial Calculation:
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)
print("Factorial:", result)


# # end
