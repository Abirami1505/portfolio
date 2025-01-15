/*Library management
Book details
recieves details about books such as id,name,genre,edition,author name*/

class BookL extends Library
{
long bookId;
String bookName;
String authorName;
String genre;
int edition;

//overriding abstract class
public void setId(long id)
{super.id=id;}
public long getId()
{return id;}

public void display()
{System.out.println("id="+id);
System.out.println("book id="+bookId);
System.out.println("book name="+bookName);
System.out.println("author name="+authorName);
System.out.println("genre="+genre);
System.out.println("edition="+edition);}

//getters and setters for bookId
public void setBookId(long bookId)
{this.bookId=bookId;}
public long getBookId()
{return bookId;}

//getters and setters for bookName
public void setBookName(String bookName)
{this.bookName=bookName;}
public String getBookName()
{return bookName;}

//getters and setters for authorName
public void setAuthorName(String authorName)
{this.authorName=authorName;}
public String getAuthorName()
{return authorName;}

//getters and setters for genre
public void setGenre(String genre)
{this.genre=genre;}
String getGenre()
{return genre;}

//getters and setters for edition
public void setEdition(int edition)
{this.edition=edition;}
public int getEdition()
{return edition;}
}