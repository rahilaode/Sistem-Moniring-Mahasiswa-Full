from monitoring import *
from tkinter import *
from tkinter import messagebox

root=Tk()
root.title('Sistem Monitoring Mahasiswa')
root.geometry('925x500+300+200')
root.configure(bg="#fff")
root.resizable(False,False)

img = PhotoImage(file="./Login.png")
Label(root, image=img, bg='white').place(x=50, y=50)

frame=Frame(root, width=350, height=350, bg="white")
frame.place(x=480, y=70)

heading=Label(frame, text="Login And Start Monitoring", fg='#57a1f8', bg='white', font=('Microsoft YaHei UI Light', 15, 'bold'))
heading.place(x=38, y=5)

#input for name
def on_enter(e):
    user=name.get()
    if user=='Nama Mahasiswa':
        name.delete(0, "end")

def on_leave(e):
    user=name.get()
    if user=='':
        name.insert(0, "Nama Mahasiswa")

name = Entry(frame, width=48,fg='black', border=0, bg='white', font=('Microsoft YaHei UI Light', 9))
name.place(x=30, y=80)
name.insert(0,'Nama Mahasiswa')
name.bind('<FocusIn>', on_enter)
name.bind('<FocusOut>', on_leave)

Frame(frame, width=300, height=2, bg='gray').place(x=25, y=100)

#input for Student ID Number
def on_enter(e):
    nim=student_id.get()
    if nim=='Nomor Induk Mahasiswa':
        student_id.delete(0, "end")

def on_leave(e):
    nim=student_id.get()
    if nim=='':
        student_id.insert(0, "Nomor Induk Mahasiswa")

student_id = Entry(frame, width=48,fg='black', border=0, bg='white', font=('Microsoft YaHei UI Light', 9))
student_id.place(x=30, y=130)
student_id.insert(0,'Nomor Induk Mahasiswa')
student_id.bind('<FocusIn>', on_enter)
student_id.bind('<FocusOut>', on_leave)

Frame(frame, width=300, height=2, bg='gray').place(x=25, y=150)

#input for class name
def on_enter(e):
    class_= class_name.get()
    if class_ == 'Mata Kuliah':
        class_name.delete(0, "end")

def on_leave(e):
    class_=class_name.get()
    if class_=='':
        class_name.insert(0, "Mata Kuliah")

class_name = Entry(frame, width=48,fg='black', border=0, bg='white', font=('Microsoft YaHei UI Light', 9))
class_name.place(x=30, y=180)
class_name.insert(0,'Mata Kuliah')
class_name.bind('<FocusIn>', on_enter)
class_name.bind('<FocusOut>', on_leave)

Frame(frame, width=300, height=2, bg='gray').place(x=25, y=200)

#messagebox
def absen():
    name_ = name.get()
    nim = student_id.get()
    matkul = class_name.get()

    if name_=="Nama Mahasiswa" or nim=="Nomor Induk Mahasiswa" or matkul=="Mata Kuliah":
        messagebox.showwarning("Peringatan", "Silahkan Lengkapi Identitas Anda !!!")
    else:
        response = messagebox.askokcancel("Pertanyaan", "Identitas anda telah direkam. Jika sudah benar, klik OK untuk mengabsen dan memulai monitoring. Apakah anda ingin lanjut ?")
        if response == 1:
            monitoring(name_,nim,matkul)

#Button
Button(frame, width=30, pady=7, text='Absen', bg='#57a1f8', fg='white', border=0, command=absen).place(x=70, y=230)
#Button(frame, width=30, pady=7, text='Mulai Monitoring', bg='#57a1f8', fg='white', border=0).place(x=70, y=280)
root.mainloop()