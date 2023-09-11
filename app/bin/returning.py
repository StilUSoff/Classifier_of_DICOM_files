import os
import shutil
import argparse
from progress.bar import IncrementalBar

class sort():
    """find and sort files"""

    def __init__(self,folder_path,folder_path_1st, folder_path_old):
        """folder path"""
        self.folder_path = folder_path
        self.folder_path_1st = folder_path_1st
        self.folder_path_old = folder_path_old
        self.massive_of_files = os.listdir(self.folder_path)

    def script(self):
        list = os.listdir(self.folder_path)
        for folder in list:
            if os.path.isdir(self.folder_path + '/' + folder):
                self.folder_path_old = self.folder_path
                self.__init__(self.folder_path + '/' + folder, self.folder_path_1st, self.folder_path_old)
                self.script()
                self.__init__(self.folder_path_old, self.folder_path_1st, self.folder_path_old)
            elif folder == '.DS_Store' or not os.path.isdir(self.folder_path + '/' + folder):
                continue
        self.sort_files(self.folder_path)

    def sort_files(self,folder_path):
        for filename in self.massive_of_files:
            if filename[0:1]=='.':
                continue
            if not os.path.isdir(filename):
                shutil.move(self.folder_path + '/' + filename, self.folder_path_1st + '/' + filename)

def runscript(folder_path):
    folder1 = sort(folder_path,folder_path,folder_path)
    folder1.script()
    list = os.listdir(folder_path)
    bar = IncrementalBar('', max=len(list))
    for files in list:
        if os.path.isdir(folder_path + '/' + files):
            if not (os.path.exists(folder_path + '/' + files) and os.listdir(folder_path + '/' + files)):
                path = os.path.join(os.path.abspath(os.path.dirname(__file__)), folder_path + '/' + files)
                shutil.rmtree(path)
        self.bar.next()
    self.bar.finish()

def main(path):
    check = 0
    while check < 11:
        check_iter = 0
        runscript(path)
        list = os.listdir(path)
        for file in list:
            if os.path.isdir(path + '/' + file) == True:
                check_iter = 1
        if check_iter == 0:
            break
        check += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs='?', type=str, help="path of your direct ( Example: /Users/Documents/My_photo )", default='test')
    args = parser.parse_args()
    main(args.path)