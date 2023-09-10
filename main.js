const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

if (process.env.NODE_ENV !== 'production') {
  app.commandLine.appendSwitch('disable-http-cache');
  app.commandLine.appendSwitch('disable-extensions');
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true, // Изменено на true
    },
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    app.quit();
  });
}

app.whenReady().then(createMainWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createMainWindow();
  }
});

ipcMain.on('folder-selected', (event, folderPath) => {
  console.log('Received folder path from renderer process:', folderPath);

  // Запускаем classifier.py с переданным путем к папке
  const pythonProcess = spawn('python', ['app/bin/classifier/classifier.py', folderPath]);

  // Обрабатываем вывод из classifier.py
  pythonProcess.stdout.on('data', (data) => {
    console.log(`classifier.py stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`classifier.py stderr: ${data}`);
  });

  // Ожидаем завершение процесса
  pythonProcess.on('close', (code) => {
    console.log(`classifier.py process exited with code ${code}`);
  });
});




// FOR package.json:
// "start": "electron main.js"