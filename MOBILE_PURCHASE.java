import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

class FlashingMessage implements Runnable{
	private JLabel l=new JLabel("");
	public FlashingMessage(JFrame f){
		f.add(l,BorderLayout.NORTH);
		new Thread(this).start();
	}
	public void run(){
		try{
			while(true){
				if(l.getText()== " "){
					l.setText("ONLINE MOBILE SHOPPING");
					Thread.sleep(500);
				}else{
					l.setText(" ");
					Thread.sleep(500);
				}
			}
		}catch(InterruptedException e){}
	}
}

class OutOfPriceRangeException extends Exception{
	OutOfPriceRangeException() {}
	public String toString(){
		return "PRODUCT NOT AVAILABLE IN THIS RANGE";
	}
}

class MOBILE_PURCHASE{
	String Brand_Name;
	String Model;
	double price;
	ArrayList<String> year = new ArrayList<String>();
	String[] brands={"Select","SAMSUNG","SONY","MOTOROLA"};
	JLabel bname, mod, prce, yr;
	JLabel warning= new JLabel();

	void resetButton(JComboBox brand, JCheckBox cb1, JCheckBox cb2, JCheckBox cb3, JTextField bnum, JTextField priceRange){
		brand.setSelectedItem("Select");
		cb1.setSelected(false);
		cb2.setSelected(false);
		cb3.setSelected(false);
		bnum.setText("");
		priceRange.setText("");
	}

	MOBILE_PURCHASE(){
		JFrame frm= new JFrame("Mobile Purchase");
		frm.setSize(300,300);
		frm.setLayout(new BorderLayout());
		frm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frm.setVisible(true);

		new FlashingMessage(frm);

		JPanel info= new JPanel();
		info.setLayout(new GridLayout(5,2));

		final JTextField bcode= new JTextField(4);

		bname= new JLabel("Brand Name: ");
		JComboBox<String> brand= new JComboBox<String>(brands);
		brand.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				String ch= (String)brand.getSelectedItem();
				if(ch=="SAMSUNG") bcode.setText("SAM");
				else if(ch=="SONY") bcode.setText("SONY");
				else if(ch=="MOTOROLA") bcode.setText("MOTO");
				else bcode.setText("");
			}
		});
		info.add(bname);
		info.add(brand);

		ItemListener il = new ItemListener(){
			public void itemStateChanged(ItemEvent ie){
				JCheckBox cb= (JCheckBox)ie.getItem();
				if(cb.isSelected())
					year.add(cb.getText());
				else
					year.remove(cb.getText());
			}
		};

		JPanel check= new JPanel();
		check.setLayout(new GridLayout(3,1));

		yr = new JLabel("Year of Product Release: ");

		JCheckBox cb1 =  new JCheckBox("2018");
		cb1.addItemListener(il);
		check.add(cb1);
		
		JCheckBox cb2= new JCheckBox("2019");
		cb2.addItemListener(il);
		check.add(cb2);

		JCheckBox cb3= new JCheckBox("2020");
		cb3.addItemListener(il);
		check.add(cb3);

		info.add(yr);
		info.add(check);

		mod = new JLabel("Model");
		JTextField bnum= new JTextField();
		bnum.addKeyListener(new KeyAdapter(){
			public void keyTyped(KeyEvent e) {
				if(bnum.getText().length()>=4)
					e.consume();
			}
		});
		bnum.setColumns(4);
		JPanel modelCode = new JPanel();
		modelCode.setLayout(new FlowLayout());
		modelCode.add(bcode);
		modelCode.add(bnum);
		
		info.add(mod);
		info.add(modelCode);

		JTextField priceRange= new JTextField();
		priceRange.addActionListener(new ActionListener(){
			//double p= Double.parseDouble(priceRange.getText());
			public void actionPerformed(ActionEvent ae){
				try{
					if(Double.parseDouble(priceRange.getText())<10000 || Double.parseDouble(priceRange.getText())>50000) throw new OutOfPriceRangeException();
				}catch(OutOfPriceRangeException e){
					warning.setText("PRODUCT NOT AVAILABLE IN THIS RANGE");
					priceRange.setText("");
				}
			}
		});
		prce = new JLabel("Price: ");
		info.add(prce);
		info.add(priceRange);

		JPanel button= new JPanel();
		button.setLayout(new GridLayout(1,2));

		JButton reset = new JButton("Reset");
		reset.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				resetButton(brand,cb1,cb2,cb3,bnum,priceRange);
			}
		});
		button.add(reset);

		JButton submit = new JButton("Submit");
		submit.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent ae){
				Brand_Name= (String)brand.getSelectedItem();
				Model= bcode.getText() + bnum.getText();
				price= Double.parseDouble(priceRange.getText());
				
				System.out.println("Brand: "+Brand_Name);
				System.out.println("Model: "+ Model);
				System.out.println("Price: "+ price);
				ListIterator li = year.listIterator();
				System.out.println("Manufacturing year: ");
				while(li.hasNext()){
					System.out.println(li.next());
				}
				resetButton(brand,cb1,cb2,cb3,bnum,priceRange);
			}
		});
		button.add(submit);
		JPanel pan=new JPanel();
		pan.setLayout(new GridLayout(2,1));
		pan.add(warning);
		pan.add(button);
		frm.add(info,BorderLayout.CENTER);
		frm.add(pan,BorderLayout.SOUTH);
	}

	public static void main(String[] args){
		SwingUtilities.invokeLater(new Runnable(){
			public void run(){
				new MOBILE_PURCHASE();
			}
		});
	}
}