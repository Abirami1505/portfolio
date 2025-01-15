/*Library Management
User details
recieves details regarding the user library card like user id,name,gender,age,valid till*/

class UserL extends Library
{
long userId;
String userName;
String gender;
int age;
String validTill;

//overriding abstract method
public void setId(long id)
{super.id=id;}
public long getId()
{return id;}

public void display()
{System.out.println("id="+id);
System.out.println("user id="+userId);
System.out.println("user name="+userName);
System.out.println("gender="+gender);
System.out.println("age="+age);
System.out.println("valid till="+validTill);}

//getters and setters for userId
public void setUserId(long userId)
{this.userId=userId;}
public long getUserId()
{return userId;}

//getters and setters for userName
public void setUserName(String userName)
{this.userName=userName;}
public String getUserName()
{return userName;}

//getters and setters for gender
public void setGender(String gender)
{this.gender=gender;}
public String getGender()
{return gender;}

//getters and setters for age
public void setAge(int age)
{this.age=age;}
public int getAge()
{return age;}

//getters and setters for validTill
public void setValidTill(String validTill)
{this.validTill=validTill;}
public String getValidTill()
{return validTill;}
}