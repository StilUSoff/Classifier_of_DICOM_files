import customtkinter as ctk
import subprocess

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Добавляем стили, аналогичные CSS из вашего HTML
        self.add_style("body", background_color="#2b2a29", font_family="Arial, sans-serif", margin=0, padding=0, display="flex", flex_direction="column", justify_content="center", align_items="center", height="100vh")
        self.add_style("h1", color="#ffffffc9", font_size="40px", font_weight=700, font_family='"Helvetica Neue", sans-serif', text_align="center", margin_top="20px")
        self.add_style("button", background_color="#440c52", color="#fff", font_size="18px", font_weight=700, font_family='"Trebuchet MS", sans-serif', padding="12px 24px", border="none", cursor="pointer", border_radius="25px", box_shadow="0px 4px 6px rgba(0, 0, 0, 0.1)", transition="background-color 0.3s, transform 0.2s", display="block", margin="0 auto")
        self.add_style("button:hover", background_color="#601174", transform="scale(1.10)")
        self.add_style("result-container", text_align="center", opacity=0, transition="opacity 0.5s ease-in-out")
        self.add_style("result-text", color="#ffffff46", font_size="20px", margin_top="20px")
        self.add_style("data-table", opacity=0, transition="opacity 0.5s ease-in-out", display="none", margin="20px auto", border_collapse="collapse", box_shadow="0px 2px 4px rgba(0, 0, 0, 0.1)", border="2px solid #440c52")
        self.add_style("th, td", padding="15px", text_align="center")
        self.add_style("th", background_color="#440c52", color="#ffffffc9")
        self.add_style("th:first-child", width="40%")
        self.add_style("th:nth-child(2)", width="30%")
        self.add_style("th:nth-child(3)", width="30%")
        self.add_style("td", background_color="#601174", color="#ffffffc9")
        self.add_style("tr:nth-child(even)", background_color="#f2f2f2")

        # Создаем виджеты, аналогичные элементам вашего HTML
        title_label = ctk.CTkLabel(self, text="Classifier of DICOM files", style="h1")
        title_label.pack()

        def select_folder():
            folder_path = ctk.askdirectory()
            if folder_path:
                command = ["python3", "classifier.py", folder_path]  # Измените путь к classifier.py на вашем компьютере
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                result_label.config(text=f"Path: {folder_path}\n{stdout.decode('utf-8')}")
                show_table()

        select_button = ctk.CTkButton(self, text="Folder for classification", style="button", command=select_folder)
        select_button.pack()

        result_label = ctk.CTkLabel(self, text="Waiting for data...", style="result-container")
        result_label.pack()

        data_table = ctk.CTkTreeview(self, style="data-table")
        data_table['columns'] = ("Name", "Modality", "Bodypart")
        data_table.heading("#1", text="Name")
        data_table.heading("#2", text="Modality")
        data_table.heading("#3", text="Bodypart")
        data_table.column("#1", width=200)
        data_table.column("#2", width=150)
        data_table.column("#3", width=150)
        data_table.pack()

        def show_table():
            data_table.pack()
            data_table["show"] = "headings"
            self.add_style("data-table", opacity=1)
            data_table.update_idletasks()

if __name__ == "__main__":
    app = App()
    app.mainloop()