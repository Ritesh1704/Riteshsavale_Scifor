#!/usr/bin/env python
# coding: utf-8

# In[12]:


class BankAccount:
    def __init__(self, account_holder, pin, balance=0):
        self.account_holder = account_holder
        self.pin = pin
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited rs{amount}. New balance: rs{self.balance}")
        else:
            print("Invalid deposit amount. Please enter a positive value.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew rs{amount}. New balance: rs{self.balance}")
        else:
            print("Invalid withdrawal amount or insufficient funds.")

    def remaining_balance(self):
        print(f"Remaining balance for {self.account_holder}: rs{self.balance}")

    def check_pin(self, entered_pin):
        return entered_pin == self.pin


# Example usage:
if __name__ == "__main__":
    # Creating a bank account for John with an initial balance of $500 and a PIN of 1234
    ritesh_account = BankAccount(account_holder="ritesh", pin="1234", balance=500)

    # Checking PIN for authentication
    entered_pin = input("Enter your PIN for authentication: ")
    if john_account.check_pin(entered_pin):
        print("Authentication successful.")
    else:
        print("Authentication failed. Please enter a valid PIN.")
   
  #deposit
    ritesh_account.deposit(200)
    
  #withdraw
    ritesh_account.withdraw(700)

