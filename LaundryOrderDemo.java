import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

class LaundryOrder{
	int orderId;
	int dressCount;
	double totalCost;
	String type;

	LaundryOrder(int orderId,int dressCount,String type){
		this.orderId= orderId;
		this.dressCount= dressCount;
		this.type= type;
	}

	void calculateOrderCost(){
		if (type=="I") totalCost= dressCount*10;
		else totalCost=dressCount*25;
	}

	void setTotalCostWithDiscount(double cost){
		double discount= cost*5/100;
		totalCost= cost-discount;
	}

	double getCost(){ return totalCost;}
}

class ScrollBanner implements Runnable{ 
	private JLabel msg= new JLabel("5% off for orders above Rs.500");
	private int x_coordinate=0;
	
	public ScrollBanner(JPanel f){
		f.add(msg);
		new Thread(this).start();
	}

	public void run(){
		while(true){
			msg.setBounds(x_coordinate,15,1000,25);
			x_coordinate++;
			if(x_coordinate==300) x_coordinate=-99;
			try{
				Thread.sleep(10);
			}catch(InterruptedException e){}
		}
	}
}

class LaundryOrderDemo{
	ArrayList<LaundryOrder> orders= new ArrayList<LaundryOrder>();
	LaundryOrder lm;
	String[] orderTypes={"Select","Iron","Wash & Iron"};
	double cost;
	
	LaundryOrderDemo(){
		JFrame frm=new JFrame("Laundry");
		frm.setSize(300,300);
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frm.setVisible(true);
		JPanel banner = new JPanel();
		banner.setLayout(new GridLayout(2,1));
		banner.add(new JLabel("Welcome to Laundry Charge Calculation Portal"));
		frm.add(banner,BorderLayout.NORTH);
		new ScrollBanner(banner);

		JPanel info = new JPanel();
		info.setLayout(new GridLayout(4,2));
		info.add(new JLabel("Order id:"));
		JTextField orderId = new JTextField();
		info.add(orderId);
		info.add(new JLabel("dress count:"));
		JTextField dressCount = new JTextField();
		info.add(dressCount);
		info.add(new JLabel("order type:"));
		JComboBox<String> type = new JComboBox<String>(orderTypes);
		info.add(type);

		JTextArea display= new JTextArea();
		JScrollPane sp=new JScrollPane(display);

		JButton submit= new JButton("Submit");
		info.add(submit);
		submit.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				String t;
				if(((String)type.getSelectedItem()).equals("Iron")) t="I";
				else if(((String)type.getSelectedItem()).equals("Wash & Iron")) t="WI";
				else{ display.setText("Enter order type"); return;}
				lm=new LaundryOrder(Integer.parseInt(orderId.getText()),Integer.parseInt(dressCount.getText()),t);
				orders.add(lm);
				lm.calculateOrderCost();
				cost=lm.getCost();
				if(cost>500){ lm.setTotalCostWithDiscount(cost);}
				display.setText("Order cost:" + String.valueOf(lm.getCost()));
			}
		});

		frm.add(info,BorderLayout.CENTER);
		frm.add(sp,BorderLayout.SOUTH);
	}

	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new LaundryOrderDemo();
			}
		});
	}
}