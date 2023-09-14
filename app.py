####### покрас CTkEntry в зеленый И ПОПЫТКА ДОБИТЬ ПРОГРЕСБАР

####### сделать главную страницу (красивое изображение по центру и перенос виджета выбора влево)


import customtkinter as ctk
import os
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import sys
sys.path.append('app/')
import classifier
sys.path.append('app/bin')
import dicom_refactor
import sort 
sys.path.append('app/bin/classifier')
import split_data
from train import MainTrain
import torch
import threading

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Classifier of DICOM files")
        self.geometry("1440x810")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=ctk.BOTH, expand=True)
        self.combobox = ctk.CTkComboBox(self.main_frame, width=210, fg_color="#601174", justify="center", values=["Classifire of images", "Sort DICOM files in path", "DICOM refactor to jpg", "Split data for model", "Train model"], command=self.combobox_callback)
        self.combobox.set("Select your algorithm")
        self.combobox.pack(pady=20)
        self.current_buttons = []
        self.save_button = None  # Initialize save_button to None
        self.bar_tr = None

    def script_callback(self, callback):
        callback()

    def combobox_callback(self, choice):
        for button in self.current_buttons:
            button.destroy()
        if not self.save_button is None:
            self.save_button.destroy()
        self.current_buttons = []
        if choice == "Classifire of images":
            self.prepare_classifier()
        if choice == "DICOM refactor to jpg":
            self.prepare_dicom_refactor()
        if choice == "Sort DICOM files in path":
            self.prepare_sort()
        if choice == "Split data for model":
            self.prepare_split_data()
        if choice == "Train model":
            self.prepare_train()

    def prepare_classifier(self):
        folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Folder for classification", command=self.select_folder_classifier)
        folder_button.pack(pady=20)
        self.check_var_cl = ctk.StringVar(value="no")
        checkbox_cl = ctk.CTkCheckBox(self.main_frame, fg_color="#601174", text="Have your own checkpoint for model?", command=self.checkbox_event_cl, variable=self.check_var_cl, onvalue="yes", offvalue="no")
        checkbox_cl.pack(pady=20)
        checkbox_cl.select()
        checkbox_cl.toggle()
        self.save_button = ctk.CTkButton(self.main_frame, fg_color="#601174", state="disabled", text="Checkpoint of your model training", command=self.select_check_classifier)  # Обновляем здесь
        self.save_button.pack(pady=20)
        self.val_button = ctk.CTkButton(self.main_frame, fg_color="#601174", state="disabled", text="val.csv file from your model trainig", command=self.select_val_classifier)  # Обновляем здесь
        self.val_button.pack(pady=20)
        self.start_cl = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Start", command=self.run_pr_cl)
        self.start_cl.pack(pady=20)
        self.current_buttons.extend([folder_button, self.save_button, self.val_button, self.start_cl, checkbox_cl])
        
    def checkbox_event_cl(self):
        choice = self.check_var_cl.get()
        if not self.save_button is None and not self.val_button is None:
            if choice == 'yes':
                self.save_button.configure(state="normal")
                self.val_button.configure(state="normal")
                self.check_save = 1
            else:
                self.save_button.configure(state="disabled")
                self.val_button.configure(state="disabled")
                self.save_button.configure(fg_color="#601174") 
                self.val_button.configure(fg_color="#601174") 
                self.check_save = 0
    def run_pr_cl(self):
        thread = threading.Thread(target=self.run_classifier, args=(self.cl_folder_path,self.check_save,))
        thread.start()

    def prepare_dicom_refactor(self):
        folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Folder for refactoring", command=self.select_folder_dicom_refactor)
        folder_button.pack(pady=20)
        start = ctk.CTkButton(self.main_frame, fg_color="#601174",text="Start", command=self.run_pr_dr)
        start.pack(pady=20)
        self.current_buttons.extend([folder_button, start])
    def run_pr_dr(self):
        thread = threading.Thread(target=self.run_dicom_refactor, args=(self.dr_folder_path,))
        thread.start()

    def prepare_sort(self):
        folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Folder for sorting", command=self.select_folder_sort)
        folder_button.pack(pady=20)
        self.check_var_body = ctk.StringVar(value="n")
        checkbox = ctk.CTkCheckBox(self.main_frame, fg_color="#601174",text="Sort by body parts?", command=self.checkbox_event_st, variable=self.check_var_body,onvalue="y", offvalue="n")
        checkbox.pack(pady=20)
        checkbox.select()
        checkbox.toggle()
        start = ctk.CTkButton(self.main_frame, fg_color="#601174",text="Start", command=self.run_pr_st)
        start.pack(pady=20)
        self.current_buttons.extend([folder_button, start, checkbox])
    def checkbox_event_st(self):
        self.sort_by_parts = self.check_var_body.get()
    def run_pr_st(self):
        thread = threading.Thread(target=self.run_sort, args=(self.sort_folder_path,self.sort_by_parts,))
        thread.start()

    def prepare_split_data(self):
        data_folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Folder with dataset", command=self.select_folder_split_data)
        data_folder_button.pack(pady=20)
        output_folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Select where to save train.csv and val.scv", command=self.select_save_split_data)
        textbox1 = ctk.CTkLabel(master=self.main_frame, justify='center', text="first row of your .csv file must be Name,Modality,Bodypart")
        textbox1.configure(state="disabled") 
        textbox1.pack(pady=20)
        output_folder_button.pack(pady=20)
        annotation_folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Path to your .scv annotation file)", command=self.select_annotation_split_data)
        annotation_folder_button.pack(pady=20)
        textbox = ctk.CTkLabel(master=self.main_frame, justify='center', text="train.csv will be used for model training, and val.csv you can use as testing of trained model")
        textbox.configure(state="disabled") 
        textbox.pack(pady=20)
        start = ctk.CTkButton(self.main_frame, fg_color="#601174",text="Start", command=self.run_pr_sp)
        start.pack(pady=20)
        self.current_buttons.extend([data_folder_button, output_folder_button, annotation_folder_button, start, textbox, textbox1])
    def run_pr_sp(self):
        thread = threading.Thread(target=self.run_split_data, args=(self.data_folder_path,self.save_data_folder_path,self.ann_data_folder_path))
        thread.start()

    def prepare_train(self):
        folder_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Path to the folder with images", command=self.select_folder_train)
        folder_button.pack(pady=20)
        textbox_csv = ctk.CTkLabel(master=self.main_frame, justify='center', text="In folder with images must be val.csv and train.csv files")
        textbox_csv.configure(state="disabled") 
        textbox_csv.pack(pady=20)
        file_button = ctk.CTkButton(self.main_frame, fg_color="#601174", text="Path to the file with attributes", command=self.select_atr_path)
        file_button.pack(pady=20)
        textbox_atr = ctk.CTkLabel(master=self.main_frame, justify='center', text="Path to your original .csv file, from wich have been created val.csv and train.csv files")
        textbox_atr.configure(state="disabled") 
        textbox_atr.pack(pady=20)
        if torch.cuda.is_available():
            self.check_var = ctk.StringVar(value="cpu")
            checkbox = ctk.CTkCheckBox(self.main_frame,fg_color="#601174", text="Work with GPU?", command=self.checkbox_event_tr, variable=self.check_var, onvalue="cuda", offvalue="cpu")
            checkbox.pack(pady=20)
            checkbox.select()
            checkbox.toggle()
        else:
            self.gpu_choice='cpu'
        self.N_epochs = ctk.CTkEntry(self.main_frame,width=180,fg_color="#601174", justify="center", placeholder_text="Enter amount of epochs")
        self.N_epochs.pack(pady=20)
        self.batch_size = ctk.CTkEntry(self.main_frame,width=290, fg_color="#601174", justify="center", placeholder_text="Enter how many samples per batch to load")
        self.batch_size.pack(pady=20)
        self.num_workers = ctk.CTkEntry(self.main_frame,width=270, fg_color="#601174", justify="center", placeholder_text="Enter how many subprocesses to use")
        self.num_workers.pack(pady=20)
        self.current_buttons.extend([folder_button, file_button, self.N_epochs, self.batch_size, self.num_workers, start, textbox_csv, textbox_atr])
        if torch.cuda.is_available():
            self.current_buttons.extend([checkbox])
        start = ctk.CTkButton(self.main_frame, fg_color="#601174",text="Start", command=self.run_pr_tr)
        start.pack(pady=20)
    def checkbox_event_tr(self):
        self.gpu_choice = self.check_var.get()
    def run_pr_tr(self):
        thread = threading.Thread(target=self.run_train)
        thread.start()


    def select_folder_classifier(self):
        self.cl_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.cl_folder_path:
            self.cl_folder_path = os.path.normpath(self.cl_folder_path)
            error = self.error_check_img(self.cl_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color="green") 

    def select_check_classifier(self):  # Создаем метод для обработки события выбора модели
        self.model_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose model checkpoint file")
        if self.model_path:
            self.model_path = os.path.normpath(self.model_path)
            error = self.error_check_clsave(self.model_path)
            if error=="Error":
                error_message = "Incorrect path: not a .pth"
                self.show_error_message(error_message)
            else:
                self.current_buttons[1].configure(fg_color="green") 

    def select_val_classifier(self):  # Создаем метод для обработки события выбора модели
        self.val_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose model checkpoint file")
        if self.val_path:
            self.val_path = os.path.normpath(self.val_path)
            error = self.error_check_csv(self.val_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[2].configure(fg_color="green")

    def select_folder_split_data(self):
        self.data_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.data_folder_path:
            self.data_folder_path = os.path.normpath(self.data_folder_path)
            error = self.error_check_img(self.data_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color="green")

    def select_save_split_data(self):
        self.save_data_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.save_data_folder_path:
            self.save_data_folder_path = os.path.normpath(self.save_data_folder_path)
            self.current_buttons[1].configure(fg_color="green") 

    def select_annotation_split_data(self):
        self.ann_data_folder_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose .csv file")
        if self.ann_data_folder_path:
            self.ann_data_folder_path = os.path.normpath(self.ann_data_folder_path)
            error = self.error_check_csv(self.ann_data_folder_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[2].configure(fg_color="green") 

    def select_folder_dicom_refactor(self):
        self.dr_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.dr_folder_path:
            self.dr_folder_path = os.path.normpath(self.dr_folder_path)
            error = self.error_check_dicom(self.dr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no DICOM files"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color="green") 

    def select_folder_sort(self):
        self.sort_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.sort_folder_path:
            self.sort_folder_path = os.path.normpath(self.sort_folder_path)
            self.current_buttons[0].configure(fg_color="green")  

    def select_folder_train(self):
        self.tr_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.tr_folder_path:
            self.tr_folder_path = os.path.normpath(self.tr_folder_path)
            error = self.error_check_img(self.tr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color="green") 

    def select_atr_path(self):
        self.tr_atr_folder_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose .csv file")
        if self.tr_atr_folder_path:
            self.tr_atr_folder_path = os.path.normpath(self.tr_atr_folder_path)
            error = self.error_check_csv(self.tr_atr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[1].configure(fg_color="green") 


    def run_classifier(self, folder_path, check):
        self.current_buttons[3].configure(fg_color="green")
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        if check == 1:
            result_lines = classifier.main(folder_path, self.model_path, self.val_path)
        else:
            result_lines = classifier.main(folder_path)
        self.data_table = MyTable(master=self.main_frame)
        self.data_table.pack(fill=ctk.BOTH, expand=True)
        self.data_table.display_table(result_lines)
        self.current_buttons.extend([self.data_table])
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        self.current_buttons[0].configure(fg_color="#601174") 

    def run_dicom_refactor(self, folder_path):
        self.current_buttons[1].configure(fg_color="green")
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        dicom_refactor.main(folder_path)
        messagebox.showinfo(title="Done!", message="You're all set")
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        self.current_buttons[0].configure(fg_color="#601174") 

    def run_sort(self, folder_path, body):
        self.current_buttons[1].configure(fg_color="green")
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        not_sorted = sort.main(folder_path,body)
        if not_sorted is None:
            messagebox.showinfo(title="Done!", message="You're all set")
        else:
            messagebox.showwarning(title="Not completely done", message="Not all of your files have been sorted, check textbox or your directory for more information")
            string = ''
            textbox = ctk.CTkTextbox(master=self.main_frame, width=700, height=400, corner_radius=2)
            textbox.insert("0.0", '\n'.join([f'{string + str(not_sorted[i])}' for i in range(len(not_sorted))]))
            textbox.configure(state="disabled") 
            textbox.pack(pady=20)
            self.current_buttons.extend([textbox])
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        self.current_buttons[0].configure(fg_color="#601174") 

    def run_split_data(self, folder_path, work_path, atribut):
        self.current_buttons[3].configure(fg_color="green")
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        result = split_data.main(folder_path, work_path, atribut)
        if result == "Error":
            messagebox.showerror(title="Error", message="Check your choices, probably incorrect .csv file")
        else:
            messagebox.showinfo(title="Done!", message="You're all set")
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        self.current_buttons[0:3].configure(fg_color="#601174") 

    def run_train(self):
        self.current_buttons[5].configure(fg_color="green")
        folder_path = self.tr_folder_path
        atr_folder_path = self.tr_atr_folder_path
        gpu_choice = self.gpu_choice
        N_epochs_wr = int(self.N_epochs.get())
        batch_size_wr = int(self.batch_size.get())
        num_workers_wr = int(self.num_workers.get())
        self.bar_tr = ctk.CTkProgressBar(master=self.main_frame, fg_color="#601174", determinate_speed=(1/float(N_epochs_wr)))
        self.bar_tr.pack(pady=20)
        self.bar_tr.set(0)
        self.current_buttons.extend([self.bar_tr])
        train_object = MainTrain(folder_path, atr_folder_path, gpu_choice, self)
        train_object.train_scipt(N_epochs=N_epochs_wr, batch_size=batch_size_wr, num_workers=num_workers_wr)
        messagebox.showinfo(title="Done!", message="You're all set")
        if hasattr(self, 'bar_tr') and self.bar_tr:
            self.bar_tr.destroy()
        self.current_buttons[0:2].configure(fg_color="#601174")

    def on_epoch_end(self):
        self.bar_tr.step()

    def error_check_img(self, work_folder):
        for filename in os.listdir(work_folder):
            if filename.endswith((".jpeg", ".tif", ".TIFF", ".tiff", ".PNG", ".JPEG", ".TIF", ".jpg",".JPG")):
                return ""
        return "Error"
    
    def error_check_dicom(self, work_folder):
        for filename in os.listdir(work_folder):
            if filename.endswith((".dcm", ".dicom", ".DCM", ".DICOM")):
                return ""
        return "Error"
    
    def error_check_csv(self, filename):
        if filename.endswith((".csv", ".CSV")):
            return ""
        return "Error"
    
    def error_check_clsave(self, filename):
        if filename.endswith((".pth", ".PTH")):
            return ""
        return "Error"
            
    def show_error_message(self, error_message):
        messagebox.showerror("Error: ", error_message)


class MyTable(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        headers = ['Name', 'Modality', 'Bodypart']
        self.tree = ttk.Treeview(self, columns=headers, show='headings', height=25)
        for header in headers:
            self.tree.heading(header, text=header)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def display_table(self, data):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for record in data:
            file_name = record.get('Name', '')
            modality = record.get('Modality', '')
            bodypart = record.get('Bodypart', '')
            self.tree.insert('', 'end', values=(file_name, modality, bodypart))


if __name__ == "__main__":
    app = App()
    app.mainloop()