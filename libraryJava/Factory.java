/*factory class of Library Management*/
class Factory
{
public static void main(String[] args)
{char ch='u';
libraryManager(ch);
System.out.println();
ch='b';
libraryManager(ch);}

static void libraryManager(char ch)
{
//book
if (ch=='b')
{BookL SASTRA=new BookL();
SASTRA.setId(1000);
SASTRA.setBookId(10001);
SASTRA.setBookName("AAA");
SASTRA.setAuthorName("aaa");
SASTRA.setGenre("Fiction");
SASTRA.setEdition(1);
SASTRA.display();}

//user
else if(ch=='u')
{UserL SASTRA=new UserL();
SASTRA.setId(2000);
SASTRA.setUserId(20001);
SASTRA.setUserName("AA");
SASTRA.setGender("Female");
SASTRA.setAge(19);
SASTRA.setValidTill("21|09|2023");
SASTRA.display();}

else
System.out.println("invalid choice");
}
} 