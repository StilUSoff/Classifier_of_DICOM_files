# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['bin\\app.py'],
    pathex=['c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin'],
    binaries=[],
    datas=[('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\app', 'app')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

a.datas += [
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\numpy', 'numpy'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\Pillow', 'Pillow'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\sklearn', 'sklearn'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\torch', 'torch'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\torchvision', 'torchvision'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\tqdm', 'tqdm'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\argparse', 'argparse'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\cv2', 'cv2'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\pydicom', 'pydicom'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\datetime', 'datetime'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\pandas', 'pandas'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\customtkinter', 'customtkinter'),
    ('c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\bin\\venv\\lib\\python3.8\\site-packages\\tensorboard', 'tensorboard'),
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Classifier app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\resources\\icon.ico'
)

app = BUNDLE(exe,
         name='Classifier app.app',
         icon='c:\\Program Files\\JetBrains\\PyCharm\\projects\\Classifier_of_DICOM_files\\resources\\icon.ico',
         bundle_identifier=None,
         version='1.0.0',
         console=False
)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='app')