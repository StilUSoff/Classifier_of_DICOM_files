const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');

function createMainWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true // Разрешение использования Node.js
    }
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    app.quit();
  });
}

app.on('ready', () => {
  createMainWindow(); // Создаем окно

  const pythonProcess = spawn('python', ['classifier.py']);

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Процесс classifier.py завершился с кодом ошибки ${code}`);
    } else {
      console.log('classifier.py успешно выполнен');
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

// Слушаем сообщение IPC с обновленными данными
ipcMain.on('update-csv-data', (event, data) => {
  // Обновляем интерфейс вашего приложения на основе данных
  // Например, можно обновить таблицу в HTML с полученными данными
  updateTable(data);
});

// Функция для обновления таблицы в HTML
// Функция для обновления таблицы в HTML
function updateTable(data) {
  // Предположим, у вас есть элемент таблицы с id "data-table"
  const tableElement = document.getElementById('data-table');

  // Очищаем таблицу перед обновлением (можете изменить логику в зависимости от ваших требований)
  tableElement.innerHTML = '';

  // Создаем заголовок таблицы
  const tableHeader = document.createElement('thead');
  const headerRow = document.createElement('tr');

  // Создаем заголовки столбцов (предположим, что у вас есть данные с заголовками)
  Object.keys(data[0]).forEach((key) => {
    const th = document.createElement('th');
    th.textContent = key;
    headerRow.appendChild(th);
  });

  tableHeader.appendChild(headerRow);
  tableElement.appendChild(tableHeader);

  // Создаем тело таблицы
  const tableBody = document.createElement('tbody');

  // Перебираем данные и создаем строки таблицы
  data.forEach((rowData) => {
    const row = document.createElement('tr');

    // Перебираем значения столбцов и создаем ячейки
    Object.values(rowData).forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    });

    tableBody.appendChild(row);
  });

  tableElement.appendChild(tableBody);
}