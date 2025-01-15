import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.time.YearMonth;

class InvalidDateException extends Exception{
	InvalidDateException(){
	}

	public String toString(){
		return "InvalidDateException: the entered date is invalid";
	}
}

class PASSENGER_DETAILS{
	String Name;
	String DOB;
	String Gender;
	String Berth;
	int Age;
	JLabel n,d,g,b,a;
	String Preferences[] = {"Upper","Middle","Lower"};
	String gen;
	JLabel senior=new JLabel();
	JLabel dateWarning=new JLabel();

	void setName(String name){
		Name=name;
	}
	void setDOB(String dob){
		DOB=dob;
	}
	void setGender(String gender){
		Gender=gender;
	}
	void setBerth(String berth){
		Berth=berth;
	}
	void setAge(int age){
		Age=age;
	}

	String display(){
		return "name: " + Name + "\nDOB: " + DOB + "\nGender: " + Gender + "\nBerth: " + Berth + "\nAge: " +Age;
	}

	PASSENGER_DETAILS(){
		JFrame frm= new JFrame();
		frm.setLayout(new BorderLayout());
		frm.setSize(300,400);
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frm.setVisible(true);
		
		JLabel heading = new JLabel("BOOKING TRAIN TICKETS");
		frm.add(heading,BorderLayout.NORTH);
		
		JPanel info = new JPanel();
		info.setLayout(new GridLayout(6,2));
		info.setSize(300,200);

		n= new JLabel("Name: ");
		d= new JLabel("DOB: ");
		g= new JLabel("Gender: ");
		b= new JLabel("Berth Preference: ");
		a= new JLabel("Age: ");

		JTextField name = new JTextField(15);
		info.add(n);
		info.add(name);

		JPanel dob= new JPanel();
		dob.setLayout(new FlowLayout());
		JTextField date= new JTextField("DD");
		JTextField month= new JTextField("MM");
		JTextField year= new JTextField("YYYY");
		dob.add(date);
		dob.add(month);
		dob.add(year);
		JTextField age= new JTextField(10);
		
		year.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				age.setText(String.valueOf(YearMonth.now().getYear()-Integer.parseInt(year.getText())));
				if(Integer.parseInt(age.getText())>60) senior= new JLabel("You are a senior citizen");
			}
		});

		info.add(d);
		info.add(dob);

		
		JPanel gender= new JPanel();
		gender.setLayout(new FlowLayout());
		JRadioButton male= new JRadioButton("Male");
		male.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				gen=male.getActionCommand();
			}
		});
		JRadioButton female= new JRadioButton("Female");
		female.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				gen=female.getActionCommand();
			}
		});
		ButtonGroup bg= new ButtonGroup();
		bg.add(male);
		bg.add(female);
		gender.add(male);
		gender.add(female);
		
		info.add(g);
		info.add(gender);

		JComboBox<String> bpref = new JComboBox<String>(Preferences);
		info.add(b);
		info.add(bpref);

		info.add(a);
		info.add(age);

		
		dob.add(dateWarning);

		JTextArea text= new JTextArea();
		JScrollPane sp= new JScrollPane(text);
		sp.setPreferredSize(new Dimension(100,50));

		JButton submit= new JButton("Submit");
		submit.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				try{
					int d= Integer.parseInt(date.getText());
					int m= Integer.parseInt(month.getText());
					if(d<1 || d>31 || m<1 || m>12){
						throw new InvalidDateException();
					}
					if (m==2){
						if(Integer.parseInt(year.getText())%4 != 0){
							if (d>28) throw new InvalidDateException();
						}
						else if(d>29){
							throw new InvalidDateException();
						}
					}
					else if(m==4||m==6||m==9||m==11){
						if (d>30) throw new InvalidDateException();
					}
					setDOB(date.getText()+"|"+month.getText()+"|"+year.getText());
					setName(name.getText());
					setGender(gen);
					String p=(String)bpref.getSelectedItem();
					setBerth(p);
					setAge(Integer.parseInt(age.getText()));
					text.setText(display());
				}catch(InvalidDateException e){
					dateWarning = new JLabel(e.toString());
					text.setText(e.toString());
				}
			}
		});	
		info.add(submit);
		frm.add(info);
		
		JPanel dis = new JPanel();
		dis.setLayout(new GridLayout(2,1));
		dis.add(senior);
		dis.add(sp);
		dis.setSize(300,200);
		frm.add(dis,BorderLayout.SOUTH);
	}


	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new PASSENGER_DETAILS();
			}
		});
	}

}