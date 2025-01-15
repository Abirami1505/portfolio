import time
import tkinter
import datetime
root=tkinter.Tk()
root.geometry("300x250")
root.title("Countdown Timer")
h=tkinter.StringVar()
m=tkinter.StringVar()
s=tkinter.StringVar()
h.set("00")
m.set("00")
s.set("00")
run=True
running=True
text=tkinter.Text(root)
text.insert(tkinter.INSERT,"\t\thh:mm:ss")
text.pack()
hentry=tkinter.Entry(root, width=3,font=("Arial",18,""),textvariable=h)
hentry.place(x=80,y=20)
mentry=tkinter.Entry(root, width=3,font=("Arial",18,""),textvariable=m)
mentry.place(x=130,y=20)
sentry=tkinter.Entry(root, width=3,font=("Arial",18,""),textvariable=s)
sentry.place(x=180,y=20)
def countdown():
     tsec=(int(h.get())*3600)+(int(m.get())*60)+int(s.get())
     while tsec>=0:
          timer=datetime.timedelta(seconds= tsec)
          timer1=str(timer)
          ftime=timer1.split(":")
          ho=ftime[0]
          mi=ftime[1]
          se=ftime[2]
          h.set("{0:2d}".format(int(ho)))
          m.set("{0:2d}".format(int(mi)))
          s.set("{0:2d}".format(int(se)))
          root.update()
          time.sleep(1)
          tsec-=1
          if running==False:
               break
          if run==False:
               break
          if tsec==0:
               text.insert(tkinter.END,"\n\n\n\n\nTime's up")
               text.pack()
def reset():
     h.set("00")
     m.set("00")
     s.set("00")
     text.delete('2.0',tkinter.END)
     global run
     run=True
def stop():
     global run
     run=False
def pause():
     global running
     running=False
     
def resume():
     global running
     running=True
     countdown()
     
start=tkinter.Button(root,text="Start",command=countdown)
start.place(x=30,y=120)
stopb=tkinter.Button(root,text="Stop",command=stop)
stopb.place(x=80,y=120)
reset=tkinter.Button(root,text="Reset",command=reset)
reset.place(x=110,y=120)
pauseb=tkinter.Button(root,text="Pause",command=pause)
pauseb.place(x=160,y=120)
resume=tkinter.Button(root,text="Resume",command=resume)
resume.place(x=200,y=120)
root.mainloop()

