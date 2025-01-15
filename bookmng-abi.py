#import modules and basic code
import sys
color = sys.stdout.shell
import mysql.connector

#menu page
def menu():
     global choice1
     print("""MENU:
****
     (+)Enter 1 for SIGN-IN
     (+)Enter 2 for SIGN-UP
     (+)Enter 3 for GUEST
     (+)Enter 4 to EXIT""")
     print()
     choice1=input("Enter your choice: ")
     print()

#sign-in, sign-up or guest
def signin():
     print()
     print("\t\t\tSIGN_IN")
     print("Enter your USERNAME and PASSWORD to sign-in")
     print()
     user=input("\t\t\tUsername: ")
     pswrd=input("\t\t\tPassword: ")
     mydb=mysql.connector.connect(host="localhost",user="root",password="abirami",database="bookmanagement")
     mycursor=mydb.cursor()
     mycursor.execute("select username,password from userpass")
     rec=mycursor.fetchall()
     if (user,pswrd) in rec:
          mycursor.execute("select shopname from userpass where username='{}'".format(user,pswrd))
          name=mycursor.fetchall()
          table1=""
          for i in name[0]:
               if i.isspace():
                    table1+="_"
               else:
                    table1+=i
          print('''Choose from menu:
                (+)1 for add data
                (+)2 for search data
                (+)3 for delete data''')
          choice2=input("Enter your choice: ")
          if choice2=="1":
               no=int(input("Enter how many data to be added: "))
               for n in range(no):
                    sno=int(input("Enter serial number: "))
                    cusname=input("Enter customer name: ")
                    bookname=input("Enter name of the book: ")
                    author=input("Enter author name: ")
                    price=input("Enter price of book: ")
                    date=input("Enter date of purchase(yyyy-mm-dd): ")
                    mycursor.execute("insert into {} values({},'{}','{}','{}',{},'{}')".format(table1,sno,cusname,bookname,author,price,date))
                    print("Data successfully added!!!")
          elif choice2=="2":
               sno=int(input("Enter serial number to display: "))
               mycursor.execute("select * from {} where sno={}".format(table,sno))
               rec1=mycursor.fetchall()
               print(rec1)
          elif choice2=="3":
               sno=int(input("Enter serial number to delete: "))
               mycursor.execute("delete from {} where sno={}".format(table,sno))
               print("Data successfully deleted!!!")
          mydb.commit()
     else:
          color.write("Username or Password incorrect\n","COMMENT")
          ans=input("Do you want to exit program?: ")
          if ans=="Y" or ans=="y" or ans=="yes" or ans=="Yes" or ans=="YES":
               print("THANK YOU!!!")
          elif ans=="N" or ans=="n" or ans=="no" or ans=="No" or ans=="NO":
               signin()
               
               
def signup():
     print("\t\t\tSIGN_UP")
     color.write("Enter shop name: ","KEYWORD")
     name=input()
     color.write("Enter username: ","KEYWORD")
     username=input()
     color.write("Enter password: ","KEYWORD")
     pswrd=input()
     color.write("Re-enter password: ","KEYWORD")
     repswrd=input()
     if pswrd==repswrd:
          mydb=mysql.connector.connect(host="localhost",user="root",password="abirami")
          mycursor=mydb.cursor()
          mycursor.execute("use bookmanagement")
          mycursor.execute("insert into userpass values('{}','{}','{}')".format(name,username,pswrd))
          color.write("Account successfully created!!!!","STRING")
          table=""
          for i in name:
               if i.isspace():
                    table+="_"
               else:
                    table+=i
          mycursor.execute("create table {}(sno int(2) primary key,customer_name varchar(20),book_name varchar(20),author varchar(20),price int(4),date_of_purchase date)".format(table))
          mydb.commit()    
     else:
          print("Passwords are not the same!")
          signup()
def guest():
     print('''Choose from menu:
                (+)1 to comment on book
                (+)2 comment on shop
                (+)3 to view comments''')
     choice3=input("Enter your choice: ")
     mydb=mysql.connector.connect(host="localhost",user="root",password="abirami",database="bookmanagement")
     mycursor=mydb.cursor()
     if choice3=="1":
          cusname=input("Enter your name (Not compulsary- Press enter if not intrested):")
          book=input("Enter book name: ")
          auth=input("Enter name of the author: ")
          comm=input("Enter your comment: ")
          mycursor.execute("insert into commentbook values('{}','{}','{}','{}')".format(cusname,book,auth,comm))
     elif choice3=="2":
          cusname=input("Enter your name (Not compulsary- Press enter if not intrested):")
          comm=input("Enter your comment: ")
          mycursor.execute("insert into commentshop values('{}','{}')".format(cusname,comm))
     elif choice3=="3":
          print('''View comments on:
                        (+)book (press 1)
                        (+)shop (press 2)''')
          choice4=input("Enter your choice: ")
          if choice4=="1":
               mycursor.execute("select * from commentbook")
               comment=mycursor.fetchall()
               print("COMMENTS:")
               print(comment)
          elif choice4=="2":
               mycursor.execute("select * from commentshop")
               comment=mycursor.fetchall()
               print("COMMENTS:")
               print(comment)
     mydb.commit()
                        
                

#main
menu()
if choice1=="1":
     signin()
elif choice1=="2":
     color.write("Are you using Book Shop management for the first time?","COMMENT")
     color.write(" Yes(Y)/No(N)","KEYWORD")
     ans=input()
     if ans=="Y" or ans=="y" or ans=="yes" or ans=="Yes" or ans=="YES":
          mydb=mysql.connector.connect(host="localhost",user="root",password="abirami")
          mycursor=mydb.cursor()
          mycursor.execute("create database bookmanagement")
          mycursor.execute("use bookmanagement")
          mycursor.execute("create table userpass(shopname varchar(20) primary key,username varchar(20) unique,password varchar(20))")
          mycursor.execute("create table commentbook(customer_name varchar(20) default 'user',bookname varchar(50),author varchar(20),comment varchar(200))")
          mycursor.execute("create table commentshop(customer_name varchar(20) default 'user',comment varchar(200))")
          mydb.commit()
          print("Welcome to book shop management!! You can now use 'BOOK SHOP MANAGEMENT PORTAL' to organise the data for your book shop...")
          signup()
     elif ans=="N" or ans=="n" or ans=="no" or ans=="No" or ans=="NO":
          print("Thank you for continueing the usage of 'BOOK SHOP MANAGEMENT PORTAL'!!!")
          signup()
elif choice1=="3":
     guest()
else:
     print("THANK YOU!!!")
     
