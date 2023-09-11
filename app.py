import customtkinter as ctk
import os
from tkinter import filedialog, messagebox
import threading
import tkinter as tk
from tkinter import ttk
import sys
sys.path.append('app/')
import classifier


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Classifier of DICOM files")
        self.geometry("800x600")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=ctk.BOTH, expand=True)
        select_button = ctk.CTkButton(self.main_frame, text="Выбрать папку", command=self.select_folder)
        select_button.pack(pady=20)
        self.data_table = MyTable(master=self.main_frame)
        self.data_table.pack(fill=ctk.BOTH, expand=True)
        

    def select_folder(self):
        folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~"), title="Choose folder")
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            error = self.error_check(folder_path)
            if error=="Error":
                error_message = "Incorrect path: no images"
                self.show_error_message(error_message)
            else:
                thread = threading.Thread(target=self.run_classifier, args=(folder_path,))
                thread.start()

    def run_classifier(self, folder_path):
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        result_lines = classifier.main(folder_path)
        self.data_table.display_table(result_lines)
        self.progress_bar.destroy()

    def error_check(self, work_folder):
        for filename in os.listdir(work_folder):
            if filename.endswith((".jpeg", ".tif", ".TIFF", ".tiff", ".PNG", ".JPEG", ".TIF", ".jpg",".JPG")):
                return ""
        return "Error"
            
    def show_error_message(self, error_message):
        # Display an error message in a dialog window
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
