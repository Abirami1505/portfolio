import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

class Free{
	int Icecream;
	int Milkshake;
}

class Food extends Free{
	HashMap<String, Integer> type= new HashMap<String, Integer>();
}

class Buyer{
	String id;
	String name;
	String address;
	double amount;
	Food food;
}

class Hotel{
	String[] foodList={"Chapati 50","Idli 20","Roast 30","Meals 20"};
	
	Hotel(){
		JFrame frm=new JFrame("Food Order");
		frm.setSize(500,400);
		frm.setLayout(new BorderLayout());
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frm.setVisible(true);
		frm.add(new JLabel("FOOD_ORDERS"),BorderLayout.NORTH);

		JPanel order= new JPanel();
		order.setLayout(new GridLayout(6,1));

		JList<String> foods=new JList<String>(foodList);
		foods.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
		order.add(new JLabel("Food: "));
		JScrollPane sp= new JScrollPane(foods);
		sp.setPreferredSize(new Dimension(150,50));

		JPanel foodPane=new JPanel();
		foodPane.setLayout(new GridLayout(1,2));
		foodPane.add(sp);
		JPanel quantity = new JPanel();
		quantity.setLayout(new GridLayout(4,2));
		quantity.add(new JLabel("chapati:"));
		JTextField cpti= new JTextField();
		cpti.setEnabled(false);
		quantity.add(cpti);
		quantity.add(new JLabel("idli:"));
		JTextField Idli= new JTextField();
		Idli.setEnabled(false);
		quantity.add(Idli);
		quantity.add(new JLabel("roast:"));
		JTextField rst= new JTextField();
		rst.setEnabled(false);
		quantity.add(rst);
		quantity.add(new JLabel("meals:"));
		JTextField mls= new JTextField();
		mls.setEnabled(false);
		quantity.add(mls);
		foodPane.add(quantity);
		order.add(foodPane);
		frm.add(order);

		foods.addListSelectionListener(new ListSelectionListener(){
			public void valueChanged(ListSelectionEvent lse){
				int idx=(int)foods.getSelectedIndex();
				if(foods.isSelectionEmpty()){
					cpti.setEnabled(false);
					Idli.setEnabled(false);
					rst.setEnabled(false);
					mls.setEnabled(false);
				}else{
					for(int i=0;i<4;i++){
					if (foods.isSelectedIndex(i)){
						if (i==0) cpti.setEnabled(true);
						else if (i==1) Idli.setEnabled(true);
						else if (i==2) rst.setEnabled(true);
						else if (i==3) mls.setEnabled(true);
					}
					else{
						if (i==0) cpti.setEnabled(false);
						else if (i==1) Idli.setEnabled(false);
						else if (i==2) rst.setEnabled(false);
						else if (i==3) mls.setEnabled(false);
					}}
				}
			}
		});

		JPanel customerDetails = new JPanel();
		customerDetails.setLayout(new GridLayout(3,2));
		order.add(new JLabel("Customer Details:"));
		customerDetails.add(new JLabel("id:"));
		JTextField id= new JTextField();
		customerDetails.add(id);
		customerDetails.add(new JLabel("name:"));
		JTextField name= new JTextField();
		customerDetails.add(name);
		customerDetails.add(new JLabel("address:"));
		JTextField address= new JTextField();
		customerDetails.add(address);
		order.add(customerDetails);

		JPanel buttons=new JPanel();
		buttons.setLayout(new GridLayout(1,2));
	}

	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new Hotel();
			}
		});
	}
}