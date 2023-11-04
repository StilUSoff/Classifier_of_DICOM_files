# DICOM File Processing Project
![Alt](project/resources/icon.png)

### This project is designed to work with medical data in DICOM format. It includes scripts and utilities for data conversion, sorting, analysis, as well as creating and training a classifier to identify the modality of images and the body parts depicted in them. Application for Windows available in [Google Drive](https://drive.google.com/file/d/1cOc_1T1I5laebfMg65PaN-KrTuxtVlvW/view?usp=drive_link).

## Description of the Initial Model

- Supports .jpg images with three RGB channels and the following modalities and body parts: CT of the brain, abdominal cavity, and chest; X-Ray of the wrist, abdominal cavity, and chest.
- However, you can train the model on your own data; all tools are provided (except for creating a .csv file with data and attributes).

## Project Structure

+ auto_py_to_exe_settings.json: Configuration file for packaging and distributing the application with auto-py-to-exe.
+ requirements.txt: File containing all the necessary Python libraries for the application.
+ project/: Directory containing scripts and processed data for the application and the application itself.
   + app.py: Python source code file for the application and its user interface.
   + venv: Python virtual environment (if used).
   + resources: Directory containing images used for packaging the application and within the application itself.
   + app/: Directory containing scripts and processed data for the application.
      + checkpoints: Folder containing saved model checkpoints.
      + logs: Model logs and journals.
      + classifier.py: Script for image classification based on the trained model.
      + bin/: Main scripts for data processing.
         + dicom_refactor.py: Script for converting DICOM files to .jpg images.
         + sort.py: Script for sorting DICOM files by modality and body parts based on metadata.
         + classifier/: Model root directory.
            + dataset.py: Script for reading attributes from .csv files and creating datasets.
            + jpg_rgb_refactor.py: Script for converting images to .jpg format with three RGB channels (choose one or both options).
            + model.py: Script containing the classifier model configuration.
            + model_test.py: Script for model testing.
            + split_data.py: Script for creating train.csv and val.csv from the .csv file with image names and attributes for training.
            + train.py: Main script responsible for model training.
            + train.csv: File created by split_data.py, containing paths and attributes of 80% of the original images for training.
            + val.csv: File created by split_data.py, containing paths and attributes of the remaining 20% of the original images for training.

## Manual Training Instructions (via Terminal)

1. Sorting DICOM Files: If you want to sort DICOM files based on metadata (e.g., Modality or BodyPartExamined), you can use sort.py. It sorts files into corresponding subdirectories within the initial directory. Run it as follows:

   ```python sort.py [path to the initial directory] [y or n]```
   - y: Sorts all files by modality and body parts.
   - n: Sorts files only by modality.

2. DICOM File Preprocessing: To preprocess DICOM files, you can use dicom_refactor.py. It converts DICOM files into JPEG images and saves them in the /img subdirectory. Run it as follows:

   ```python dicom_refactor.py [path to the directory with DICOM files]```

3. Converting Images to RGB: If you want to convert images to RGB format, you can use jpg_rgb_refactor.py. It performs this operation and saves the updated files in the same directory. You can run it as follows:

   ```python jpg_rgb_refactor.py [path to the directory with images] [0 or 1 or 2]```

   - 0: Convert to JPEG format only.
   - 1: Convert to RGB format only.
   - 2: Both steps (JPEG and RGB).

4. Data Preparation for Model Training: Before starting the model training for modality and body part recognition, you should split the data into training and testing using split_data.py. This script creates train.csv and val.csv files containing image names, modality, and body part information. However, this information is taken from a user-provided .csv file with labels. You can run it as follows:

   ```python split_data.py [path to the initial directory with images] [path to the working directory where train.csv and val.csv will be saved] [path to the .csv file containing image information]```


5. Model Training: If you have prepared all the necessary data for model training, use train.py. This script uses all the data you've prepared to train the image classifier. You can run it as follows:

   ```python split_data.py [path to the directory with images] [path to train.py] [path where model checkpoints will be saved after training] [use CPU for training - "cpu", use GPU for training - "cuda"] [number of training epochs (more epochs = longer training time and increased accuracy)] [number of images loaded into memory at once (smaller batch size = longer training time and increased accuracy)] [number of processes concurrently generating batches (more parallel processes = less training time and higher CPU load)]```


## Environment Recommendations

- Create a Python virtual environment to isolate project dependencies.
- Install all required Python libraries by running the command ```python3 -m pip install -r requirements.txt```
- If you want to recreate the application build process (or modify it), you can run the following commands in the terminal:
   - ```python -m venv venv```
   - ```source venv/bin/activate``` (or ```venv\Scripts\activate``` for Windows)
   - ```pip install -r requirements.txt```

## Requirements

- Python 3.11.xx
- Python libraries specified in requirements.txt
- If model training does not use CUDA but your system configuration meets the requirements, it is recommended to use the command ```pip install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html```
