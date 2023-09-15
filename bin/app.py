from PIL import Image
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
import webbrowser
import traceback

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.main_color = "#006900"
        self.hover_color = "#00b500"
        self.chosen_color = "#003400"
        ctk.set_appearance_mode("dark")
        self.title("Classifier of DICOM files")
        self.geometry("1440x810")
        self.minsize(820, 680)
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=ctk.BOTH)
        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.pack(fill=ctk.BOTH, expand=True)
        self.combobox = ctk.CTkOptionMenu(self.main_frame, fg_color=self.main_color, button_color = self.main_color, button_hover_color=self.hover_color, dropdown_fg_color=self.main_color, dropdown_hover_color=self.hover_color, width=80, values=["Classifire of images", "Sort DICOM files in path", "DICOM refactor to jpg", "Split data for model", "Train model"], command=self.combobox_callback)
        self.combobox.set("Select your algorithm")
        self.combobox.pack(pady=20, padx=20, side='left', anchor="nw")
        home = ctk.CTkImage(Image.open("resources/home_button.png"), size=(16,16))
        self.home_button = ctk.CTkButton(self.main_frame, hover_color = self.hover_color,fg_color = self.main_color, height=30, width=30, image=home, command=self.main_page_back, text="")
        self.home_button.pack(pady=20, padx=20, side='right', anchor="ne")
        self.current_buttons = []
        self.bar_tr = None
        self.main_page()

    def combobox_callback(self, choice):
        for button in self.current_buttons:
            button.destroy()
        if hasattr(self, 'data_table') and self.data_table:
            self.data_table.destroy()
        if hasattr(self, 'textbox') and self.textbox:
            self.textbox.destroy()
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

    def main_page_back(self):
        for button in self.current_buttons:
            button.destroy()
        self.current_buttons = []
        self.main_page()

    def main_page(self):
        img = ctk.CTkImage(Image.open("resources/icon.png"), size=(256, 256))
        image_label = ctk.CTkLabel(self.center_frame, image=img, text="")
        image_label.pack(pady=150)
        github_link = ctk.CTkLabel(self.center_frame, text="GitHub Repository", text_color=self.main_color, cursor="hand2")
        github_link.pack(pady=10, side='bottom')
        github_link.bind("<Button-1>", lambda event: self.github_open())
        self.current_buttons.extend([image_label, github_link])
    def github_open(self):
        webbrowser.open("https://github.com/StilUSoff/Classifier_of_DICOM_files")

    def prepare_classifier(self):
        folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Folder for classification", command=self.select_folder_classifier)
        folder_button.pack(pady=20)
        self.check_var_cl = ctk.StringVar(value="no")
        checkbox_cl = ctk.CTkCheckBox(self.center_frame, fg_color=self.main_color, text="Have your own checkpoint for model?", command=self.checkbox_event_cl, variable=self.check_var_cl, onvalue="yes", offvalue="no")
        checkbox_cl.pack(pady=10)
        self.save_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, state="disabled", text="Checkpoint of your model training", command=self.select_check_classifier)  # Обновляем здесь
        self.save_button.pack(pady=20)
        self.val_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, state="disabled", text="val.csv file from your model trainig", command=self.select_val_classifier)  # Обновляем здесь
        self.val_button.pack(pady=20)
        checkbox_cl.select()
        checkbox_cl.toggle()
        self.start_cl = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Start", command=self.run_pr_cl)
        self.start_cl.pack(pady=20)
        self.current_buttons.extend([folder_button, self.start_cl, self.save_button, self.val_button, checkbox_cl])
        
    def checkbox_event_cl(self):
        choice = self.check_var_cl.get()
        if choice == 'yes':
            self.save_button.configure(state="normal")
            self.val_button.configure(state="normal")
            self.check_save = 1
        else:
            self.save_button.configure(state="disabled")
            self.val_button.configure(state="disabled")
            self.save_button.configure(fg_color=self.main_color) 
            self.val_button.configure(fg_color=self.main_color) 
            self.check_save = 0
    def run_pr_cl(self):
        thread = threading.Thread(target=self.run_classifier, args=(self.cl_folder_path,self.check_save,))
        thread.start()

    def prepare_sort(self):
        folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Folder for sorting", command=self.select_folder_sort)
        folder_button.pack(pady=20)
        self.check_var_body = ctk.StringVar(value="n")
        checkbox = ctk.CTkCheckBox(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color,text="Sort by body parts?", command=self.checkbox_event_st, variable=self.check_var_body,onvalue="y", offvalue="n")
        checkbox.pack(pady=10)
        checkbox.select()
        checkbox.toggle()
        start = ctk.CTkButton(self.center_frame,hover_color=self.hover_color, fg_color=self.main_color,text="Start", command=self.run_pr_st)
        start.pack(pady=20)
        self.current_buttons.extend([folder_button, start, checkbox])
    def checkbox_event_st(self):
        self.sort_by_parts = self.check_var_body.get()
    def run_pr_st(self):
        thread = threading.Thread(target=self.run_sort, args=(self.sort_folder_path,self.sort_by_parts,))
        thread.start()

    def prepare_dicom_refactor(self):
        folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Folder for refactoring", command=self.select_folder_dicom_refactor)
        folder_button.pack(pady=20)
        start = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color,text="Start", command=self.run_pr_dr)
        start.pack(pady=20)
        self.current_buttons.extend([folder_button, start])
    def run_pr_dr(self):
        thread = threading.Thread(target=self.run_dicom_refactor, args=(self.dr_folder_path,))
        thread.start()

    def prepare_split_data(self):
        data_folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Folder with dataset", command=self.select_folder_split_data)
        data_folder_button.pack(pady=20)
        output_folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Select where to save train.csv and val.scv", command=self.select_save_split_data)
        textbox1 = ctk.CTkLabel(master=self.center_frame, justify='center', text="first row of your .csv file must be Name,Modality,Bodypart")
        textbox1.configure(state="disabled") 
        textbox1.pack(pady=5)
        output_folder_button.pack(pady=20)
        annotation_folder_button = ctk.CTkButton(self.center_frame,hover_color=self.hover_color, fg_color=self.main_color, text="Path to your .scv annotation file", command=self.select_annotation_split_data)
        annotation_folder_button.pack(pady=20)
        textbox = ctk.CTkLabel(master=self.center_frame, justify='center', text="train.csv will be used for model training, and val.csv you can use as testing of trained model")
        textbox.configure(state="disabled") 
        textbox.pack(pady=5)
        start = ctk.CTkButton(self.center_frame,hover_color=self.hover_color, fg_color=self.main_color,text="Start", command=self.run_pr_sp)
        start.pack(pady=20)
        self.current_buttons.extend([data_folder_button, output_folder_button, annotation_folder_button, start, textbox, textbox1])
    def run_pr_sp(self):
        thread = threading.Thread(target=self.run_split_data, args=(self.data_folder_path,self.save_data_folder_path,self.ann_data_folder_path))
        thread.start()

    def prepare_train(self):
        folder_button = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Path to the folder with images", command=self.select_folder_train)
        folder_button.pack(pady=20)
        textbox_csv = ctk.CTkLabel(master=self.center_frame,justify='center', text="In folder with images must be val.csv and train.csv files")
        textbox_csv.configure(state="disabled") 
        textbox_csv.pack(pady=5)
        file_button = ctk.CTkButton(self.center_frame,hover_color=self.hover_color, fg_color=self.main_color, text="Path to the file with attributes", command=self.select_atr_path)
        file_button.pack(pady=20)
        save_checkpoints = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color, text="Path to the directory where checkpoints of train will be saving", command=self.select_check_save)
        save_checkpoints.pack(pady=20)
        textbox_atr = ctk.CTkLabel(master=self.center_frame,justify='center', text="Path to your original .csv file, from wich have been created val.csv and train.csv files")
        textbox_atr.configure(state="disabled") 
        textbox_atr.pack(pady=5)
        if torch.cuda.is_available():
            self.check_var = ctk.StringVar(value="cpu")
            checkbox = ctk.CTkCheckBox(self.center_frame,fg_color=self.main_color, text="Work with GPU?", command=self.checkbox_event_tr, variable=self.check_var, onvalue="cuda", offvalue="cpu")
            checkbox.pack(pady=10)
            checkbox.select()
            checkbox.toggle()
        else:
            self.gpu_choice='cpu'
        self.N_epochs = ctk.CTkButton(self.center_frame,hover_color=self.hover_color,fg_color=self.main_color,text="Amount of epochs", command=self.N_epochs_click_event)
        self.N_epochs.pack(padx=20, pady=20)
        self.batch_size = ctk.CTkButton(self.center_frame,hover_color=self.hover_color,fg_color=self.main_color, text="Samples per batch", command=self.batch_size_click_event)
        self.batch_size.pack(padx=20, pady=20)
        self.num_workers = ctk.CTkButton(self.center_frame,hover_color=self.hover_color,fg_color=self.main_color, text="Subprocesses", command=self.num_workers_click_event)
        self.num_workers.pack(padx=20, pady=20)
        self.start_ts = ctk.CTkButton(self.center_frame, hover_color=self.hover_color,fg_color=self.main_color,text="Start", command=self.run_pr_tr)
        self.start_ts.pack(pady=20)
        self.current_buttons.extend([folder_button, file_button, save_checkpoints, self.N_epochs, self.batch_size, self.num_workers, self.start_ts, textbox_csv, textbox_atr])
        if torch.cuda.is_available():
            self.current_buttons.extend([checkbox])

    def N_epochs_click_event(self):
        self.N_epochs_wr = ctk.CTkInputDialog(text="Type in a number of epochs for train:", title="Amount of epochs").get_input()
        if self.N_epochs_wr:
            if self.N_epochs_wr.isnumeric():
                self.current_buttons[3].configure(fg_color=self.chosen_color) 
            else:
                error_message = "Incorrect iput: only numerics"
                self.show_error_message(error_message)
                
    def batch_size_click_event(self):
        self.batch_size_wr = ctk.CTkInputDialog(text="Type in a number of samples per batch to load:", title="Samples per batch").get_input()
        if self.batch_size_wr:
            if self.batch_size_wr.isnumeric():
                self.current_buttons[4].configure(fg_color=self.chosen_color) 
            else:
                error_message = "Incorrect iput: only numerics"
                self.show_error_message(error_message)

    def num_workers_click_event(self):
        self.num_workers_wr = ctk.CTkInputDialog(text="Type in a number of subprocesses to use:", title="Subprocesses").get_input()
        if self.num_workers_wr:
            if self.num_workers_wr.isnumeric():
                self.current_buttons[5].configure(fg_color=self.chosen_color) 
            else:
                error_message = "Incorrect iput: only numerics"
                self.show_error_message(error_message)

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
                self.current_buttons[0].configure(fg_color=self.chosen_color) 

    def select_check_classifier(self):  # Создаем метод для обработки события выбора модели
        self.model_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose model checkpoint file")
        if self.model_path:
            self.model_path = os.path.normpath(self.model_path)
            error = self.error_check_clsave(self.model_path)
            if error=="Error":
                error_message = "Incorrect path: not a .pth"
                self.show_error_message(error_message)
            else:
                self.current_buttons[2].configure(fg_color=self.chosen_color) 

    def select_val_classifier(self):  # Создаем метод для обработки события выбора модели
        self.val_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose model checkpoint file")
        if self.val_path:
            self.val_path = os.path.normpath(self.val_path)
            error = self.error_check_csv(self.val_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[3].configure(fg_color=self.chosen_color)

    def select_folder_split_data(self):
        self.data_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.data_folder_path:
            self.data_folder_path = os.path.normpath(self.data_folder_path)
            error = self.error_check_img(self.data_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color=self.chosen_color)

    def select_save_split_data(self):
        self.save_data_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.save_data_folder_path:
            self.save_data_folder_path = os.path.normpath(self.save_data_folder_path)
            self.current_buttons[1].configure(fg_color=self.chosen_color) 

    def select_annotation_split_data(self):
        self.ann_data_folder_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose .csv file")
        if self.ann_data_folder_path:
            self.ann_data_folder_path = os.path.normpath(self.ann_data_folder_path)
            error = self.error_check_csv(self.ann_data_folder_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[2].configure(fg_color=self.chosen_color) 

    def select_folder_dicom_refactor(self):
        self.dr_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.dr_folder_path:
            self.dr_folder_path = os.path.normpath(self.dr_folder_path)
            error = self.error_check_dicom(self.dr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no DICOM files"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color=self.chosen_color) 

    def select_folder_sort(self):
        self.sort_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.sort_folder_path:
            self.sort_folder_path = os.path.normpath(self.sort_folder_path)
            self.current_buttons[0].configure(fg_color=self.chosen_color)  

    def select_folder_train(self):
        self.tr_folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.tr_folder_path:
            self.tr_folder_path = os.path.normpath(self.tr_folder_path)
            error = self.error_check_img(self.tr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                self.current_buttons[0].configure(fg_color=self.chosen_color) 

    def select_atr_path(self):
        self.tr_atr_folder_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Choose .csv file")
        if self.tr_atr_folder_path:
            self.tr_atr_folder_path = os.path.normpath(self.tr_atr_folder_path)
            error = self.error_check_csv(self.tr_atr_folder_path)
            if error=="Error":
                error_message = "Incorrect path: not a .csv"
                self.show_error_message(error_message)
            else:
                self.current_buttons[1].configure(fg_color=self.chosen_color) 
    
    def select_check_save(self):
        self.check_saves = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if self.check_saves:
            self.check_saves = os.path.normpath(self.check_saves)
            self.current_buttons[2].configure(fg_color=self.chosen_color)  


    def run_classifier(self, folder_path, check):
        self.start_cl.configure(fg_color=self.chosen_color)
        self.progress_bar = ctk.CTkProgressBar(self.center_frame, fg_color=self.main_color, progress_color=self.hover_color,mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        try:
            if check == 1:
                result_lines = classifier.main(folder_path, self.model_path, self.val_path)
            else:
                result_lines = classifier.main(folder_path)
            if hasattr(self, 'data_table') and self.data_table:
                self.data_table.destroy()
            self.data_table = MyTable(master=self.center_frame)
            self.data_table.pack(fill=ctk.BOTH, expand=True)
            self.data_table.display_table(result_lines)
            messagebox.showinfo(title="Done!", message="You're all set")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback.print_exc()  # Эта строка выводит трассировку стека в консоль для дополнительной информации
            self.show_error_message(error_message)
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        if check==1:
            [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:4]]
        else:
            [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:2]]

    def run_sort(self, folder_path, body):
        self.current_buttons[1].configure(fg_color=self.chosen_color)
        self.progress_bar = ctk.CTkProgressBar(self.center_frame, fg_color=self.main_color, progress_color=self.hover_color,mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        try:
            not_sorted = sort.main(folder_path,body)
            if not_sorted is None:
                messagebox.showinfo(title="Done!", message="You're all set")
            else:
                messagebox.showwarning(title="Not completely done", message="Not all of your files have been sorted, check textbox or your directory for more information")
                if hasattr(self, 'textbox') and self.textbox:
                    self.textbox.destroy()
                string = ''
                self.textbox = ctk.CTkTextbox(master=self.center_frame, width=700, height=400, corner_radius=2)
                self.textbox.insert("0.0", '\n'.join([f'{string + str(not_sorted[i])}' for i in range(len(not_sorted))]))
                self.textbox.configure(state="disabled") 
                self.textbox.pack(pady=20)
                self.current_buttons.extend([self.textbox])
                messagebox.showinfo(title="Done!", message="You're all set")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback.print_exc()  # Эта строка выводит трассировку стека в консоль для дополнительной информации
            self.show_error_message(error_message)
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:2]]

    def run_dicom_refactor(self, folder_path):
        self.current_buttons[1].configure(fg_color=self.chosen_color)
        self.progress_bar = ctk.CTkProgressBar(self.center_frame, fg_color=self.main_color, progress_color=self.hover_color,mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        try:
            dicom_refactor.main(folder_path)
            messagebox.showinfo(title="Done!", message="You're all set")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback.print_exc()  # Эта строка выводит трассировку стека в консоль для дополнительной информации
            self.show_error_message(error_message)
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:2]]            

    def run_split_data(self, folder_path, work_path, atribut):
        self.current_buttons[3].configure(fg_color=self.chosen_color)
        self.progress_bar = ctk.CTkProgressBar(self.center_frame, fg_color=self.main_color, progress_color=self.hover_color,mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        try:
            result = split_data.main(folder_path, work_path, atribut)
            if result == "Error":
                messagebox.showerror(title="Error", message="Check your choices, probably incorrect .csv file")
            else:
                messagebox.showinfo(title="Done!", message="You're all set")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback.print_exc()  # Эта строка выводит трассировку стека в консоль для дополнительной информации
            self.show_error_message(error_message)
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.destroy()
        [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:4]]

    def run_train(self):
        self.start_ts.configure(fg_color=self.chosen_color)
        folder_path = self.tr_folder_path
        atr_folder_path = self.tr_atr_folder_path
        check_saves = self.check_saves
        gpu_choice = self.gpu_choice
        self.bar_tr = ctk.CTkProgressBar(master=self.center_frame, fg_color=self.main_color, progress_color=self.hover_color, determinate_speed=(50/float(self.N_epochs_wr)))
        self.bar_tr.pack(pady=20)
        self.bar_tr.set(0)
        self.current_buttons.extend([self.bar_tr])
        try:
            train_object = MainTrain(folder_path, atr_folder_path, gpu_choice, self)
            train_object.train_scipt(checkpoint=check_saves, N_epochs=int(self.N_epochs_wr), batch_size=int(self.batch_size_wr), num_workers=int(self.num_workers_wr))
            messagebox.showinfo(title="Done!", message="You're all set")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback.print_exc()  # Эта строка выводит трассировку стека в консоль для дополнительной информации
            self.show_error_message(error_message)
        if hasattr(self, 'bar_tr') and self.bar_tr:
            self.bar_tr.destroy()
        [button.configure(fg_color=self.main_color) for button in self.current_buttons[0:7]]

    def on_epoch_end(self):
        self.bar_tr.step()
        print("step")

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