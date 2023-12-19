#!/usr/bin/env python
# coding: utf-8

# # oops

# # Abstraction
# 
# - Abstraction is the process of hiding the implementation details and showing only the functionality to the user
# - In Python, we can achieve abstraction by using abstract classes and methods
# - An abstract class is a class that contains one or more abstract methods, which are methods that are declared but not defined
# - To create an abstract class, we need to import the abc module and use the @abstractmethod decorator
# - An abstract method must be overridden by the subclasses that inherit from the abstract class

# In[8]:


from abc import ABC, abstractmethod


# In[20]:


from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

circle = Circle(5)
print(circle.area()) 


# In[1]:


from abc import ABC ,abstractmethod
class a(ABC):
    @abstractmethod
    def area(self):pass
class square(a):
    def __init__(self,side):
        self.side=side
    def area(self):
        return 4*self.side
Square=square(5)
Square.area()


# Explanation:
#     Importing Necessary Modules:
# 
# from abc import ABC, abstractmethod: This line imports the ABC (Abstract Base Class) and abstractmethod from the abc module. These are used for defining abstract classes and methods.
# Defining Abstract Class:
# 
# class a(ABC):: This line defines an abstract class named oo that inherits from ABC, making it an abstract base class.
# @abstractmethod: This decorator is used to declare an abstract method named area within the abstract class. Any concrete subclass of oo must provide an implementation for this method.
# Defining Concrete Subclass:
# 
# class square(a):: This line defines a concrete subclass named square that inherits from the abstract class a.
# def __init__(self, side):: This is the constructor method that initializes an instance of the square class with a side attribute.
# def area(self):: This method provides a concrete implementation of the abstract area method declared in the abstract class oo. It calculates the area of a square using the formula 4 * side.
# Creating an Instance:
# 
# Square = square(5): This line creates an instance of the square class with a side length of 5.
# Calculating and Printing the Area:
# 
# result = Square.area(): This line calls the area method on the Square instance, which calculates the area using the concrete implementation in the square class.
# print(result): The result of the area calculation (in this case, 20) is printed to the console.
# So, the code defines an abstract class a with an abstract method area, provides a concrete implementation of the area method in the square subclass, creates an instance of square with a side length of 5, calculates and prints the area, which is 20.
