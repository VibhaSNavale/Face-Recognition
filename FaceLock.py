import tkinter
import tkinter as tk
import tkinter.messagebox
from tkinter import *
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import webbrowser

creds = "new1.txt"

s= tkinter.Tk()

def faceUnlock():
    
    def end():
        cap.release()
        cv2.destroyAllWindows()
    
    data_path = 'D:/faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    #print("Model Training Complete!!!!!")
    
    face_classifier = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi
    
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

            
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
        

            if result[1] < 600:
                confidence = int(100*(1-(result[1])/300))

            if confidence >= 80:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (10, 255, 100), 2)
                cv2.putText(image, "Press ENTER to Continue", (55,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,200,245), 2)
                cv2.imshow('Face Cropper', image)
                if cv2.waitKey(1)==13:
                    cap.release()
                    cv2.destroyAllWindows()
                    login()

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "Press ESC to Exit", (75, 50), cv2.FONT_HERSHEY_COMPLEX, 1,  (255, 212, 59), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, "Press ESC to Exit", (140, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 212, 59), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==27: 
            end()
            

def Quit():
    s.destroy()
  
    
def loginpage():
    s.update()
    s.deiconify()
    s.title("Login Details")
    s.geometry("650x450")

    label ='LOGIN DETAILS:'
    tk.Label(s, text = label,font='Helvetica',fg='black',width=35,height=3).grid(row=0,column=1)
    
    tkinter.Label(s, text="Username:",font='Helvetica').grid(row=1)
    g1 = tkinter.Entry(s)
    g1.grid(row=1, column=1,ipadx=20,ipady=2)
    
    tk.Label(s).grid(row=2)    
    
    tkinter.Label(s, text="Password:",font='Helvetica').grid(row=3)
    g2 = tkinter.Entry(s,show='*')
    g2.grid(row=3, column=1,ipadx=20,ipady=2)
    
    def checklogin():
        with open(creds) as f:
            l=0
            for j in f.readlines():
                l=l+1
        with open(creds) as f:
            data = f.readlines()
            for i in range(1,l):
                if(g1.get()==data[i].rstrip()):                                    
                    if(g2.get()==data[i+1].rstrip()):
                        print("Welcome",data[i+2])
                        semester()
                        return
                    else:
                        print("invalid password")
                        return
                i+=4
            print("invalid username")
           
    tk.Label(s).grid(row=4)
    tkinter.Button(s, text ="Face Unlock", command = faceUnlock).grid(row=5,column=0,ipadx=20)
    tkinter.Button(s, text ="Login", command = checklogin).grid(row=5,column=1,ipadx=20)
    tkinter.Button(s, text ="Signup ", command = reset).grid(row=5,column=2,ipadx=20)
    tk.Label(s).grid(row=6)
    tkinter.Button(s, text ="Reset", command = loginpage).grid(row=7,column=0,ipadx=20)
    tkinter.Button(s, text ="Exit", command = Quit).grid(row=7,column=2,ipadx=20)
    
    s.mainloop()

  
def reset():
    s.withdraw()
    m= tkinter.Tk()
    m.title("Login Details")
    m.geometry("500x500")
    
    def Exit():
        m.destroy()
        s.destroy()
        
        
    label ='Enter your Login Details:'
    tk.Label(m, text = label,font='Helvetica',fg='black',width=25,height=2).grid(row=0,column=1)

    tk.Label(m, text="First Name:",font='Helvetica').grid(row=1)
    e1 = tkinter.Entry(m)
    e1.grid(row=1,column=1,ipadx=20,ipady=2)
    
    tkinter.Label(m, text="Last Name:",font='Helvetica').grid(row=2)
    e2 = tkinter.Entry(m)
    e2.grid(row=2,column=1,ipadx=20,ipady=2)
    
    tkinter.Label(m, text="Username:",font='Helvetica').grid(row=3)
    e3 = tkinter.Entry(m)
    e3.grid(row=3,column=1,ipadx=20,ipady=2)
    
    tkinter.Label(m, text="Password:",font='Helvetica').grid(row=5)
    e4 = tkinter.Entry(m,show='*')
    e4.grid(row=5,column=1,ipadx=20,ipady=2)
    
    def signup():
        
        with open(creds,'a') as f:
            f.write("\n")
            f.write(e3.get())
            f.write("\n")
            f.write(e4.get())
            f.write("\n")
            f.write(e1.get())        
            f.write("\n")
            f.write(e2.get())        
            f.close()
        m.destroy()
        semester()

    
    def addFace():
        
        face_classifier = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        
        def face_extractor(img):
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            if faces is():
                return None

            for(x,y,w,h) in faces:
                cropped_face = img[y:y+h, x:x+w]

            return cropped_face

        cap = cv2.VideoCapture(0)
        count = 0
        
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count+=1
                face = cv2.resize(face_extractor(frame),(200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = 'D:/faces/'+e3.get()+"."+str(count)+'.jpg'
                cv2.imwrite(file_name_path,face)

                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper',face)
            else:
                print("Face not found")
                pass

            if cv2.waitKey(1)==13 or count==50:
                break
        cap.release()
        cv2.destroyAllWindows()
        
        print('Collecting Samples Complete!!!')
        
        data_path = 'D:/faces/'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

        Training_Data, Labels = [], []

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

        Labels = np.asarray(Labels, dtype=np.int32)

        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(np.asarray(Training_Data), np.asarray(Labels))      
#        print("Model Training Complete!!!!!")
    def back():
        m.destroy()
        loginpage()
        
    def resetSignup():
        m.withdraw()
        reset()
        
        
        
        
    tk.Label(m).grid(row=5)
    tk.Label(m).grid(row=6)
    tkinter.Button(m, text ="Add face lock ", command = addFace).grid(row=7,column=0,ipadx=20)
    tkinter.Button(m, text ="Sign Up", command = signup).grid(row=7,column=1,ipadx=20)
    tkinter.Button(m, text ="Reset ", command = resetSignup).grid(row=7,column=2,ipadx=20)
    tk.Label(m).grid(row=8)
    tkinter.Button(m, text ="Back", command = back).grid(row=9,column=0,ipadx=25)
    tkinter.Button(m, text ="Quit", command = Exit).grid(row=9,column=2,ipadx=25)
 
    m.mainloop()


def semester():
    
    v=tkinter.Tk()
    
    hi="Login Successful!\n\nSelect Semester:"
    lbl = Label(v,text = hi,font='Verdana 15')  
                
    v.title("Login Successful")

    s.withdraw()
    v.geometry("350x350")
    
    def logout():
        v.destroy()
        loginpage()
        
    def sem3():
    
        sub3=tkinter.Tk()
        v.withdraw()
        sub3.geometry("350x250")
        sub3.title("Semester 3")

        lbl = Label(sub3,text = "Select subject:",font='Verdana 15')  
    
        listbox3 = Listbox(sub3, width=100, height=50)  
   
        listbox3.insert(1,"1: Analog and Digital Electronics ") 
        listbox3.insert(2,"2: Discrete Mathematical Structures") 
        listbox3.insert(3,"3: Data Structres") 
        listbox3.insert(4,"4: Computer Organization") 
        listbox3.insert(5,"5: UNIX And Shell Programming") 
        listbox3.insert(6,"6: Mathematics-III") 
      
        def choose():
            sel=listbox3.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/ADE-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/ADE-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

            elif(sel==(1,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/dms-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/dms-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/DSC-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/DSC-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/CO=1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/CO-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(4,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/UN-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/UN-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(5,)):
                i=tkinter.Tk()
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/math-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem3/math-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

        def cancelsem():
            sub3.destroy()
            v.deiconify()
            v.destroy()
            semester()
   
        def logout():
            sub3.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub3)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)
        
        lbl.pack()  
        listbox3.pack()
        sub3.mainloop()
       
    def sem4():
    
        sub4=tkinter.Tk()
        v.withdraw()
        sub4.geometry("350x250")
        sub4.title("Semester 4")

        lbl = Label(sub4,text = "Select subject:",font='Helvetica')  
    
        listbox4 = Listbox(sub4, width=100, height=50)  
   
        listbox4.insert(1,"1: Mathematics-IV ") 
        listbox4.insert(2,"2: Microprocessor and Microcontroller") 
        listbox4.insert(3,"3: Software Engineering") 
        listbox4.insert(4,"4: Design and Analysis Of Algorithms") 
        listbox4.insert(5,"5: Object Oriented Programming with JAVA") 
        listbox4.insert(6,"6: Data Communications") 
                
        def choose():
            sel=listbox4.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/math-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/math-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(1,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/MP-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/MP-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/SE-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/SE-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/DAA-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/DAA-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(4,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/OOP-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/OOP-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(5,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/DC-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem4/DC-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

        def cancelsem():
            sub4.destroy()
            v.deiconify()
            v.destroy()
            semester()
        
        def logout():
            sub4.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub4)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)

        lbl.pack()  
        listbox4.pack()
        sub4.mainloop()
    
    def sem5():
    
        sub5=tkinter.Tk()
        v.withdraw()
        sub5.geometry("350x250")
        sub5.title("Semester 5")
        
        lbl = Label(sub5,text = "Select subject:",font='Helvetica')  
        
        listbox5 = Listbox(sub5, width=100, height=50)  
        
        listbox5.insert(1,"1: Management and Enterpreneurship For IT Industry ") 
        listbox5.insert(2,"2: Computer Networks-I") 
        listbox5.insert(3,"3: Database Management System") 
        listbox5.insert(4,"4: Automata Theory And Computability") 
        listbox5.insert(5,"5: Advanced JAVA and J2EE") 
        listbox5.insert(6,"6: Artificial Intelligence") 
        listbox5.insert(7,"7: Cloud Computing")
                
        def choose():
            sel=listbox5.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/MnE-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/MnE-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

            elif(sel==(1,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/CN-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/CN-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/DMS-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/DMS-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/ATC-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/ATC-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(4,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/AJ-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/AJ-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(5,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/AI-1.pdf')
                    i.destroy()
                 
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
                
            elif(sel==(6,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem5/CC-1.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
                
                i.mainloop()

        def cancelsem():
            sub5.destroy()
            v.deiconify()
            v.destroy()
            semester()
            
        def logout():
            sub5.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub5)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)
        
        lbl.pack()  
        listbox5.pack()
        sub5.mainloop()
    
    def sem6():
    
        sub6=tkinter.Tk()
        v.withdraw()
        sub6.geometry("350x250")
        sub6.title("Semester 6")
        
        lbl = Label(sub6,text = "Select subject:",font='Helvetica')  
        
        listbox6 = Listbox(sub6, width=100, height=50)  
   
        listbox6.insert(1,"1: Management and Enterpreneurship ") 
        listbox6.insert(2,"2: Computer Networks-II") 
        listbox6.insert(3,"3: Operations Research") 
        listbox6.insert(4,"4: Computer Graphics And Visualization") 
        listbox6.insert(5,"5: UNIX System Programming") 
        listbox6.insert(6,"6: Programming Languages") 
           
        def choose():
            sel=listbox6.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/MnE-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/MnE-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(1,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/CN-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/CN-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/OR-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/OR-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/GnV.pdf')
                    i.destroy()
                       
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
                   
                i.mainloop()
                
            elif(sel==(4,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/UP-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/UP-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(5,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/PL-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem6/PL-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

        def cancelsem():
            sub6.destroy()
            v.deiconify()
            v.destroy()
            semester()
        
        def logout():
            sub6.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub6)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)

        lbl.pack()  
        listbox6.pack()
        sub6.mainloop()
    
    def sem7():
    
        sub7=tkinter.Tk()
        v.withdraw()
        sub7.geometry("350x250")
        sub7.title("Semester 7")

        lbl = Label(sub7,text = "Select subject:",font='Helvetica')  
    
        listbox7 = Listbox(sub7, width=100, height=50)  
   
        listbox7.insert(1,"1: Object Oriented Modeling And Design ") 
        listbox7.insert(2,"2: Programming the web") 
        listbox7.insert(3,"3: JAVA and J2EE") 
        listbox7.insert(4,"4: C# Programming And .Net") 
        listbox7.insert(5,"5: Storage Area Networks") 
           
        def choose():
            sel=listbox7.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/OOM-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/OOM-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
              
            elif(sel==(1,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/WP-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/WP-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/Java-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/Java-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/net-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/net-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(4,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/SAN-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem7/SAN-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()

        def cancelsem():
            sub7.destroy()
            v.deiconify()
            v.destroy()
            semester()
       
        def logout():
            sub7.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub7)
        buttonFrame.pack(side=BOTTOM)
        
        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)

        lbl.pack()  
        listbox7.pack()
        sub7.mainloop()
        
    def sem8():
    
        sub8=tkinter.Tk()
        v.withdraw()
        sub8.geometry("350x250")
        sub8.title("Semester 8")

        lbl = Label(sub8,text = "Select subject:",font='Helvetica')  
    
        listbox8 = Listbox(sub8, width=100, height=50)  
        
        listbox8.insert(1,"1: System Modeling And Simulation ") 
        listbox8.insert(2,"2: Software Architectures") 
        listbox8.insert(3,"3: Information And Network Security") 
        listbox8.insert(4,"4: Information Retrieval") 
            
        def choose():
            sel=listbox8.curselection()
            if(sel==(0,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/SMS-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/SMS-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
             
            elif(sel==(1,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('180x170')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/sa-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/sa-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(2,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/ins-1.pdf')
                    i.destroy()
                
                def paper2():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/ins-2.pdf')
                    i.destroy()
                
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)
        
                cancelButton = Button(i, text="Question paper 2 (JAN 2018)", command=paper2).grid(row=1,column=0)
                
                i.mainloop()
                
            elif(sel==(3,)):
                i=tkinter.Tk()
                i.title("Question Papers")
                i.geometry('200x200')
                def paper1():
                    webbrowser.open_new(r'file:///C:/Users/user/Desktop/QP/ise/sem8/ir-1.pdf')
                    i.destroy()
                chooseButton = Button(i, text="Question paper 1 (JUNE 2017)", command=paper1).grid(row=0,column=0)

                i.mainloop()
        
        def cancelsem():
            sub8.destroy()
            v.deiconify()
            v.destroy()
            semester()
            
        def logout():
            sub8.destroy()
            v.destroy()
            loginpage()

        buttonFrame = Frame(sub8)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
        cancelButton = Button(buttonFrame, text="Back", command=cancelsem).grid(row=0,column=2,ipadx=15)
        
        cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=4,ipadx=15)

        lbl.pack()  
        listbox8.pack()
        sub8.mainloop()
                
    listbox = Listbox(v)  
   
    listbox.insert(1," Sem III") 
    listbox.insert(2," Sem IV") 
    listbox.insert(3," Sem V") 
    listbox.insert(4," Sem VI") 
    listbox.insert(5," Sem VII") 
    listbox.insert(6," Sem VIII") 
          
    def choose():
        sel=listbox.curselection()
        if(sel==(0,)):
            sem3()
        elif(sel==(1,)):
            sem4()
        elif(sel==(2,)):
            sem5()
        elif(sel==(3,)):
            sem6()
        elif(sel==(4,)):
            sem7()
        elif(sel==(5,)):
            sem8()           
        
    buttonFrame = Frame(v)
    buttonFrame.pack(side=BOTTOM)
        
    chooseButton = Button(buttonFrame, text="Choose", command=choose).grid(row=0,column=0,ipadx=15)
        
    cancelButton1 = Button(buttonFrame, text="LogOut", command=logout).grid(row=0,column=2,ipadx=15)

    lbl.pack()  
    listbox.pack(side="left",fill="both",expand=True)
    v.mainloop()  

def login():
    semester()      

loginpage()