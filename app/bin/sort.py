import os
import pydicom
import shutil
import argparse
from progress.bar import IncrementalBar

class sort():
    """find and sort files"""

    def __init__(self,folder_path):
        """folder path"""
        self.folder_path = folder_path
        self.massive_of_files = os.listdir(self.folder_path)
        self.bar = IncrementalBar ('', max = len(os.listdir(self.folder_path)))

    def script(self,body):
        """main script"""
        check = 0
        self.sort_files(check)
        if body == 'y':
            check = 1
            self.__init__(self.folder_path)
            for folder in self.massive_of_files:
                if folder == '.DS_Store':
                    continue
                elif os.path.exists(self.folder_path + '/' + folder):
                    self.__init__(self.folder_path + '/' + folder)
                    self.sort_files(check)
                    old_folder_path = self.folder_path.replace('/' + folder, '')
                    self.__init__(old_folder_path)
                self.bar.next()
        self.bar.finish()

    def sort_files(self, check):
        """sort of files (check==0: sort by modality; check==1: sort by body parts)"""
        for filename in self.massive_of_files:
            if filename[0:1]=='.':
                continue
            metadata = self.metadata_find(filename)
            if metadata==0:
                continue

            if check==0:
                full_path = self.folder_path + '/' + metadata.get((0x0008, 0x0060))[0:]
            else:
                full_path = self.folder_path + '/' + metadata.get((0x0018, 0x0015))[0:]

            if not os.path.exists(full_path):
                os.mkdir(full_path)
            shutil.move(self.file_path, full_path)

    def metadata_find(self, filename):
        """initialization of metadata"""
        self.filename = filename
        if self.filename == '.DS_Store' or os.path.isdir(self.filename):
            return 0
        self.file_path = self.folder_path + '/' + self.filename
        metadata = pydicom.dcmread(self.file_path, force=True)
        return metadata

def main(path,body):
    folder1 = sort(path)
    folder1.script(body)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs='?', type=str, help="path of your direct ( Example: /Users/Documents/My_photo )", default='test')
    parser.add_argument("body", nargs='?', type=str, help="here you choose to sort by body parts or do not ( y OR n )", default='y')
    args = parser.parse_args()
    main(args.path, args.body)