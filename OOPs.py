#!/usr/bin/env python
# coding: utf-8

# # Python OOPs Concepts

# ### Object - Oriented Programming

# In[1]:


x =10 #x is an object reference and 10 is your object


# In[2]:


id(x)


# In[3]:


hex(id(x))


# - In Python, everything is an object
# - A class is the blueprint for the object

# ![image.png](attachment:image.png)
# 
# 
# `credit to the creator`

# - Cars such as Audi, BMW, Lamborghini etc. -- all these are differnt car companies are objects of the class `car`
# - Object is an instance of a class
# - An object has following two characteristics - Attribute & Behaviour
#     * name, price, color as atrributes (**variables**)
#     * acceleration, speed, braking etc. as **behaviour (methods/functions)**

# #### Let's define a class in Python

# - Creating a `Person` class with the name, gender, and profession instance variables

# In[20]:


class Person:
    def __init__(self, name, gender, profession):
        #data members (instance variables) 
        self.name = name
        self.gender = gender
        self.profession = profession
        
    #Behavior (instance methods)
    def show(self):
        print('Name:', self.name, 'Gender:', self.gender, 'Profession:', self.profession)
    
    def work(self):
        print(self.name, 'is working as a', self.profession)
     


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[22]:


# Create object of a class
person1 = Person('Akash', 'Male', 'Lead Data Scientist')


#Call methods
person1.show()
person1.work()


# In[24]:


# Create object of a class
person2 = Person('Pallavi', 'Female', 'Educator')


#Call methods
person2.show()
person2.work()


# ### Drawing similarities between built-in class (list) and our class (Person)

# In[15]:


my_list = [10,20,30, 50,20,50,20,40,20]


# In[16]:


type(my_list)


# In[17]:


dir(list)


# In[18]:


my_list.count(20) #Return number of occurrences of value


# `list`: `class` => **Person**
# 
# `my_list` : `object` => **person1**
# 
# `count()`: `method / function` => **show(), work()**
# 

# In[25]:


import os
os.getcwd()


# ![image.png](attachment:image.png)

# In Object-oriented programming, Inside a Class, we can define the following three types of methods.
# 
# - Instance method: Used to access or modify the object state. If we use instance variables inside a method, such methods are called instance methods.
# 
# - Class method: Used to access or modify the class state. In method implementation, if we use only class variables, then such type of methods we should declare as a class method.
# 
# - Static method: It is a general utility method that performs a task in isolation. Inside this method, we don’t use instance or class variable because this static method doesn’t have access to the class attributes.

# ### Methods example

# In[27]:


class Student:
    #class variable
    school_name = "St. Paul's High School"
    
    
    #constructor
    def __init__(self, name, age):
        #instance variables
        self.name = name
        self.age =age
        
    
    #instance method
    def show(self):
        #access instance variables and class variables
        print('Student:', self.name, 'having age:', self.age, 'studied in', Student.school_name)
        
    #instance method
    def change_age(self, new_age):
        #modify the instance variable
        self.age = new_age
        
    #class method
    @classmethod #is a decorator; used to define a class method. Class methods are bound to the class and not the instance of the class
    def modify_school_name(cls, new_name):
        #modify class variable
        cls.school_name = new_name #modifies the class variable 'school_name' by assigning it a new value 'new_name' 
                                   #'cls' here refers to the class itself, allowing the method to access and mofify class attributes


# In[28]:


#Create an object 'student1`
student1 = Student("Rutuja", 20)

#call instance methods
student1.show()
student1.change_age(18)


# In[29]:


student1.show()


# In[30]:


#call class method
Student.modify_school_name('XYZ School')

#call instance method
student1.show()


# #### Delete Object 

# - In Python, we can delete the object by using `del` keyword

# In[22]:


del student1


# ## Encapsulation in Python

# - Encapsulation is one of the fundamental concepts in OOPs like inheritance, polymorphism. 
# - It describes the concept of building data and methods within a single unit
# - A class is an example of encapsulation as it binds all the data members (instance variables) and methods into a single unit

# ![image.png](attachment:image.png)

# - Prevents outer classes from accessing and changing attributes and methods of a class
# - Basically helps in `data shielding`

# In[31]:


class Employee:
    
    def __init__(self, name, project):
        #instance variables
        self.name = name
        self.project = project
        
    #instance method
    def work(self):
        print(self.name, 'is working on', self.project)
        


# ![image.png](attachment:image.png)

# ### Public Member

# In[36]:


class Employee:
    
    def __init__(self, name, salary, project): #double underscore before and after init
        #instance variables / data members
        self.name = name
        self.salary = salary
        self.project = project
        
    #instance method to display employee's details:
    def show(self):
        #accessing public data member
        print('Name:',self.name, 'is having salary:', self.salary)
        
    def work(self):
        print(self.name, 'is working on the project:', self.project)
        


# In[37]:


### Create an object using class Employee
emp101 = Employee('Akash', 100000, 'BlueMirror')


# In[38]:


### calling public methods of the class
emp101.show()


# In[39]:


### calling public methods of the class
emp101.work()


# ### Access Modifiers in Python

# - Python provides three types of access modifiers:
#     1. Public
#     2. Protected
#     3. Private

# In[ ]:


class Employee:
    
    def __init__(self, name, salary, project): #double underscore before and after init
        #instance variables / data members
        self.name = name #public data member (no underscores)
        self._project = project  #protected data member (single underscore)
        self.__salary = salary   #private data member (double underscore)


# **In Python, we dont have direct access modifiers hence we can achieve this by using `single` or `double` underscores**

# ![image.png](attachment:image.png)

# ### Private member

# **Private members are accessible only within the class, and we cant access them directly from the class objects**

# In[55]:


class Employee_new:
    
    def __init__(self, name, salary, project): #double underscore before and after init
        #instance variables / data members
        self.name = name #public data member (no underscores)
        self._project = project  #protected data member (single underscore)
        self.__salary = salary   #private data member (double underscore)


# In[51]:


### create an object of Employee class:
emp102 = Employee_new('Rohan', 150000, 'Rockers')


# In[52]:


### Access public data members
print('Salary', emp101.salary) #emp101 ==> Akash (this is coming from clas Employee)


# In[53]:


### Access private data members
print('Salary', emp102.salary) #emp102 ==> Rohan (coming from new class Employee_new)


# **Using `name mangling`** to access private member

# ### Protected member

# In[54]:


class Employee_new:
    
    def __init__(self, name, salary, project): #double underscore before and after init
        #instance variables / data members
        self.name = name #public data member (no underscores)
        self._project = project  #protected data member (single underscore)
        self.__salary = salary   #private data member (double underscore)


# ## Inheritance in Python

# The **process of inheriting the properties of the `parent class into a child class` is called inheritance**.

# The existing class is called a base class or parent class and the new class is called a subclass or child class or derived class

# - Main purpose of inheritance is the `reusability` of code.
# - Since we are using an existing class to create a new class instead of creating it from scratch

# - For example, in the real world, **Car is a sub class of a Vehicle Class**

# In Python, based upon the number of child and parent classes involved, there are five types of inheritance. The type of inheritance are listed below:
# 
# 1. Single inheritance
# 2. Multiple Inheritance
# 3. Multilevel inheritance
# 4. Hierarchical Inheritance
# 5. Hybrid Inheritance

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# #### Single Inheritance

# In[56]:


### Base / Parent Class
class Vehicle:
    def vehicle_info(self):
        print('Vehicle is the parent class')
        
#Child Class
class Car(Vehicle):
    def car_info(self):
        print('This is the child class: Car')


# In[57]:


### Create object of Car which is the child class
audi = Car()


# In[67]:


###Access Vehicle's info using car(audi) object
audi.car_info()
audi.vehicle_info()


# In[60]:


car101 = Vehicle()


# In[61]:


car101.car_info()


# #### Multiple Inheritance

# In[64]:


### Parent Class 1
class Person:
    def person_info(self, name, age):
        print('This is the parent class 1: Person')
        print('Name:', name, '| Age:', age)
    
### Parent Class 2
class Company:
    def company_info(self, company_name, location):
        print('This is the parent class 2: Company')
        print('Company name:', company_name, '| Company Location:', location)
        
        
### Child Class
class Employee(Person, Company):
    def employee_info(self, salary, skill):
        print('This is child class: Employee')
        print('Salary:', salary, '| Skill:', skill)
    


# In[66]:


#Create an object of employee => child class
emp105 = Employee()

#access data
emp105.person_info('Akash', 34) #parent class 1
emp105.company_info('Accenture Strategy & Consulting', 'Bangalore') #parent class 2
emp105.employee_info(25000, 'Lead Data Scientist') #child class


# `Observation`: Here we created two parent classes namely: `Person and Company`. And then we created a child class called Employee which `inherited` Person and Company classes info

# #### Multilevel Inheritance (chaining classes)

# In[71]:


#Base class
class Vehicle:
    def vehicle_info(self):
        print('This is the parent /super class:Vehicle')

#Child Clas --> Level 0
class Car(Vehicle):
    def car_info(self):
        print('This is the Level 0 Child Class: Car')
        
#child class --> Level 1
class SportsCar(Car):
    def sportscar_info(self):
        print('This is Level 1 Child Class: Sports Car')
        

#child class --> Level 2
class Mclaren(SportsCar):
    def mclaren_info(self):
        print('This is Level 2 Mclaren sports car Child Class: Mclaren')
               


# In[73]:


### Create an oject of Mclaren Sportscar models
mclaren_p1 = Mclaren()

###access Vehicle's info and car info and sportscar info using mclaren object
mclaren_p1.vehicle_info()
mclaren_p1.car_info()
mclaren_p1.sportscar_info()
mclaren_p1.mclaren_info()


# ### H/W Reading assignments
#     1. Try illustrating rest of inheritances
#     2. Python super() function

# #### Hierarchical Inheritance

# In[1]:


#base class

class Vehicle:
    def vehicle_info(self):
        print('This is the parent class: Vehicle')

#child class #1
class Car(Vehicle):
    def car_info(self):
        print('This is level 0 child class 1: Car')
        
#child class #2
class Truck(Vehicle):
    def truck_info(self):
        print('This is level 0 child class 2: Truck')
        
        
#child class #3
class Van(Vehicle):
    def van_info(self):
        print('This is level 0 child class 3: Van')


# In[2]:


obj1 = Car()


# In[3]:


obj1.car_info()


# In[4]:


obj1.vehicle_info()


# In[5]:


obj1.truck_info()


# In[6]:


obj2 = Truck()


# In[7]:


obj2.truck_info()


# In[8]:


obj2.vehicle_info()


# ### H/W Hybrid Inheritance

# ### Python super() function

# - In any child class, we can refer to the parent class by using the `super()` function in child class
# - We are not required to remember or specify the parent class name to access its methods
# - The primary purpose of `super()` function is to enable inheritance and facilitate method overloading

# ### super() with single inheritance

# In[42]:


#parent class
class Company:
    def company_name(self):
        return 'Accenture'

#child class
class Employee(Company):
    def info(self):
        #calling the super class method using super() function
        c_name = super().company_name()
        print('Akash works at', c_name)


# In[43]:


#create an object of child class
emp111 = Employee()


# In[25]:


emp111.company_name()


# In[36]:


emp111.info()


# #### Method overloading - Read about this more

# In[56]:


#parent class
class Company:
    def company_name(self):
        print('This is company_name from the parent class')

#child class
class Employee(Company):
    def company_name(self):
        super().company_name()
        print('This is an extended behavior in the child class')    


# In[57]:


emp707 = Employee()


# In[58]:


emp707.company_name()


# - Whenever a method with the same name exists in both the parent and child, the child class can use `super()` function to invoke the parent class and then extend or modify its behavior

# ## Polymorphism in Python

# `poly` means `many` and `morph` means `forms`

# ![image.png](attachment:image.png)

# - For example: Sakshi acts as an employee when she is in the office
# - When she is at home, she acts like a wife
# - Basically, Sakshi represents herself differently in different places

# **In Polymorphism, a method can process objects differently depending on the `class type` or `data type`**

# #### Polymorphism in Built-in function `len()`

# In[37]:


students = ['Sakshi', 'Aman', 'Abhishek']
school_name = 'XYZ School'


# In[38]:


print(len(students)) #counts the number of items/elements in the list


# In[39]:


print(len(school_name)) #count the number of characters including whitespaces in the variable


# In[40]:


print(type(students))


# In[41]:


print(type(school_name))


# The built-in function len() calculates the length of an object depending upon its type. 
# 
# If an object is a string, it returns the count of characters, and 
# 
# If an object is a list, it returns the count of items in a list.

# In[ ]:




