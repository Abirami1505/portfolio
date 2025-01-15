import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

class Appointment{
	String repName;
	String whomToMeet;
	String appointmentDate;

	Appointment(String repName,String whomToMeet,String appointmentDate){
		this.repName=repName;
		this.whomToMeet=whomToMeet;
		this.appointmentDate=appointmentDate;
	}
}

class OrganizedAppointment{
	JTextField rname,meet,adate;
	JLabel r,m,a;

	ArrayList<Appointment> appointment= new ArrayList<Appointment>();
	Appointment temp;
	String searchRep(String rep){
		ListIterator li= appointment.listIterator();
		while(li.hasNext()){
			temp=(Appointment)li.next();
			if(rep.equals(temp.repName)){
				return "whom to meet: "+ temp.whomToMeet + "\nappointment date" + temp.appointmentDate;
			}
		}
		return "The given representative name is not in the list";
	}

	String searchDate(String date){
		ListIterator li= appointment.listIterator();
		while(li.hasNext()){
			temp=(Appointment)li.next();
			if(date.equals(temp.appointmentDate)){
				return "representative name: " + temp.repName + "\nwhom to meet: "+ temp.whomToMeet ;
			}
		}
		return "The given date is not in the list";
	}

	void clear(JTextField a, JTextField b, JTextField c){
		a.setText("");
		b.setText("");
		c.setText("");
	}

	OrganizedAppointment(){
		CardLayout cl = new CardLayout();
		JFrame frm= new JFrame("Appointment Scheduler");
		frm.setSize(300,300);
		frm.setLayout(new BorderLayout());
		frm.setVisible(true);
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		JPanel mainPanel= new JPanel();
		mainPanel.setLayout(cl);

		//add
		JPanel addWindow = new JPanel();
		addWindow.setLayout(new GridLayout(4,2));
		r= new JLabel("representative name:");
		rname=new JTextField();
		addWindow.add(r);
		addWindow.add(rname);
		m= new JLabel("whom to meet:");
		meet=new JTextField();
		addWindow.add(m);
		addWindow.add(meet);
		a= new JLabel("Appointment date:");
		adate=new JTextField();
		addWindow.add(a);
		addWindow.add(adate);
		JButton submit1 = new JButton("Submit");
		submit1.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				Appointment temp= new Appointment(rname.getText(),meet.getText(),adate.getText());
				appointment.add(temp);
				clear(rname,meet,adate);
			}
		});
		addWindow.add(submit1);
		mainPanel.add(addWindow,"Addition");

		//search
		JPanel searchWindow = new JPanel();
		searchWindow.setLayout(new GridLayout(4,1));
		JButton name= new JButton("Search with Rep Name");
		JButton date= new JButton("Search with appointment date");
		searchWindow.add(name);
		searchWindow.add(date);

		JTextArea display= new JTextArea();
		JScrollPane sp = new JScrollPane(display);
		sp.setPreferredSize(new Dimension(100,50));

		JPanel search= new JPanel();
		search.setLayout(cl);

		//name search
		JPanel searchName =new JPanel();
		searchName.setLayout(new GridLayout(2,2));
		searchName.add(new JLabel("Enter Name:"));
		JTextField nameSearch= new JTextField();
		searchName.add(nameSearch);
		JButton submit2= new JButton("Submit");
		submit2.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				display.setText(searchRep(nameSearch.getText()));
			}
		});
		searchName.add(submit2);

		//date search
		JPanel searchDate =new JPanel();
		searchDate.setLayout(new GridLayout(2,2));
		searchDate.add(new JLabel("Enter Date:"));
		JTextField dateSearch= new JTextField();
		searchDate.add(dateSearch);
		JButton submit3= new JButton("Submit");
		submit3.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				display.setText(searchDate(dateSearch.getText()));
			}
		});
		searchDate.add(submit3);
		
		search.add(searchName,"Name");
		search.add(searchDate,"Date");

		name.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(search,"Name");
				display.setText("");
			}
		});

		date.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(search,"Date");
				display.setText("");
			}
		});

		searchWindow.add(search);
		searchWindow.add(sp);

		mainPanel.add(searchWindow,"Search");

		frm.add(mainPanel);

		JLabel head= new JLabel("Appointment Scheduler");
		JMenuBar jmb=new JMenuBar();
		JMenu main= new JMenu("Main");
		JMenuItem Add= new JMenuItem("Add");
		JMenuItem Search= new JMenuItem("Search");
		main.add(Add);
		main.add(Search);
		jmb.add(main);
		frm.setJMenuBar(jmb);
		frm.add(head,BorderLayout.NORTH); 

		Add.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(mainPanel,"Addition");
			}
		});

		Search.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(mainPanel,"Search");
			}
		});
	}

	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new OrganizedAppointment();
			}
		});
	}
}
