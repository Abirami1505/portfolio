import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

class VehicleNotFoundException extends Exception{
	VehicleNotFoundException(){
		System.out.println("Vehicle not found exception");
	}
	
	public String toString(){
		return "VEHICLE NOT FOUND";
	}
}


class Vehicle{
	int regNo;
	String vehType;
	String vehMake;
	int contNo;
	double cost;

	Vehicle(int regNo,String vehType,String vehMake,int contNo,double cost){
		this.regNo= regNo;
		this.vehType= vehType;
		this.vehMake= vehMake;
		this.contNo= contNo;
		this.cost= cost;
	}
	
	int getRegNo(){
		return regNo;
	}

	double getCost(){
		return cost;
	}

	public String toString(){
		return "["+regNo+ " | " + vehType + " | " + vehMake + " | " + contNo + " | " + cost + "]";
	}  
}

class SortServiceCost implements Comparator<Vehicle>{
	public int compare(Vehicle v1,Vehicle v2){
		Double cost1=v1.getCost();
		Double cost2=v2.getCost();
		int value = cost1.compareTo(cost2);
		if (value>0) return 1;
		else if (value<0) return -1;
		else return 0;
	}
}

class ServiceManagement{
	TreeMap<Integer,Vehicle> data= new TreeMap<Integer,Vehicle>();
	int count=0; boolean flag = false;
	Vehicle temp;

	void addVehicle(Vehicle v){
		count+=1;
		data.put(count,v);
	}

	double getServiceCost(int rNo) throws VehicleNotFoundException{
		for (int idx: data.keySet()){
			temp= data.get(idx);
			if (temp.getRegNo() == rNo){
				flag=true;
				break;
			}
		}
		if (flag==false){
			throw new VehicleNotFoundException();
		}
		return temp.getCost();
	}

	ArrayList<Vehicle> listAllVehicles(){
		ArrayList<Vehicle> veh = new ArrayList<Vehicle>();
		for (int i: data.keySet()){
			veh.add(data.get(i));
		}
		Collections.sort(veh, new SortServiceCost());
		return veh;	
	}
}

class VehicleManager{
	ServiceManagement sm = new ServiceManagement();
	JLabel rno,vtyp,vmk,cno,scst;
	String[] VType={"Select","Two Wheeler","Four wheeler"};
	int regNo;
	String vehType;
	String vehMake;
	int contNo;
	double cost;

	String msg="";
	
	VehicleManager(){
		JFrame frm = new JFrame("Vehicle Service Center");
		frm.setSize(300,300);
		frm.setLayout(new BorderLayout());
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	
		JLabel head= new JLabel("Vehicle Service Manager");
		frm.add(head,BorderLayout.NORTH);

		CardLayout cl = new CardLayout();

	//main window for card layout
		JPanel main= new JPanel();
		main.setLayout(cl);

	//add window
		JPanel addVehicle = new JPanel();
		addVehicle.setLayout(new GridLayout(6,2));

		//regno
		rno = new JLabel("Register Number:");
		addVehicle.add(rno);
		JTextField regno = new JTextField(15);
		addVehicle.add(regno);

		//vehtype
		vtyp = new JLabel("Vehicle Type:");
		addVehicle.add(vtyp);
		JComboBox<String> typ = new JComboBox<String>(VType);
		addVehicle.add(typ);

		//vehmake
		vmk = new JLabel("Vehicle Make:");
		addVehicle.add(vmk);
		JTextField vmak = new JTextField(15);
		addVehicle.add(vmak);

		//conNo
		cno = new JLabel("Contact Number:");
		addVehicle.add(cno);
		JTextField conno = new JTextField();
		addVehicle.add(conno);

		//sercost
		scst = new JLabel("Service Cost:");
		addVehicle.add(scst);
		JTextField sercost = new JTextField();
		addVehicle.add(sercost);

		//add button
		JButton Add = new JButton("Add Vehicle");
		Add.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				regNo= Integer.parseInt(regno.getText());
				vehType= (String)typ.getSelectedItem();
				vehMake= vmak.getText();
				contNo= Integer.parseInt(conno.getText());
				cost= Double.parseDouble(sercost.getText());
				sm.addVehicle(new Vehicle(regNo,vehType,vehMake,contNo,cost));
				regno.setText("");
				typ.setSelectedItem("Select");
				vmak.setText("");
				conno.setText("");
				sercost.setText("");
			}
		});
		addVehicle.add(Add);

	//service cost window
		JPanel getServiceCost = new JPanel();
		getServiceCost.setLayout(new GridLayout(2,1));

		JTextArea display = new JTextArea();
		JScrollPane sp = new JScrollPane(display);

		//info pane
		JPanel getInfo= new JPanel();
		getInfo.setLayout(new GridLayout(2,2));

		//reg no
		getInfo.add(new JLabel("Enter Registration no.:"));
		JTextField regEnt = new JTextField();
		getInfo.add(regEnt);

		//search button
		JButton SerCost = new JButton("Get Service Cost");
		SerCost.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				regNo= Integer.parseInt(regEnt.getText());
				try{
					cost=sm.getServiceCost(regNo);
					display.setText("Service Cost: " + String.valueOf(cost));
				}catch(VehicleNotFoundException e){
					display.setText(e.toString());
				}
			}
		});
		getInfo.add(SerCost);

		getServiceCost.add(getInfo);
		getServiceCost.add(sp);

	//display window
		JPanel listVehicle = new JPanel();
		listVehicle.setLayout(new BorderLayout());
		JTextArea list = new JTextArea();
		JScrollPane sp2 = new JScrollPane(list);
		
		JButton listveh = new JButton("List vehicle");
		listveh.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				ArrayList<Vehicle> v= sm.listAllVehicles();
				ListIterator li= v.listIterator();
				while(li.hasNext()){
					msg+=li.next()+"\n";
				}
				list.setText(msg);
			}
		});
		listVehicle.add(listveh,BorderLayout.NORTH);
		listVehicle.add(sp2,BorderLayout.CENTER);

		main.add(addVehicle,"Add");
		main.add(getServiceCost,"Search");
		main.add(listVehicle,"View");

	//menu bar
		JMenuBar mb = new JMenuBar();

		JMenu menu1 = new JMenu("Menu");

		JMenuItem AddData = new JMenuItem("Add Vehicle");
		JMenuItem SearchData = new JMenuItem("View Service Cost");
		JMenuItem View = new JMenuItem("View all data");
		
		menu1.add(AddData);
		menu1.add(SearchData);
		menu1.add(View);

		mb.add(menu1);

		frm.setJMenuBar(mb);

		frm.add(main,BorderLayout.CENTER);
		frm.setVisible(true);

		AddData.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(main,"Add");
			}
		});
		SearchData.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(main,"Search");
			}
		});
		View.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				cl.show(main,"View");
			}
		});

		
	}

	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new VehicleManager();
			}
		});
	}
}