const { ipcRenderer, contextBridge } = require('electron');

// Используем contextBridge для безопасного взаимодействия с главным процессом
contextBridge.exposeInMainWorld('electron', {
  sendToMain: (channel, data) => {
    ipcRenderer.send(channel, data);
  },
});

document.addEventListener('DOMContentLoaded', function () {
  // Этот код выполняется после полной загрузки DOM

  document.getElementById('myButton').addEventListener('click', function () {
    const inputElement = document.createElement('input');
    inputElement.type = 'file';
    inputElement.webkitdirectory = true;
    inputElement.addEventListener('change', function (event) {
      const folderPath = event.target.files[0].path;
      updateResult(`Path: ${folderPath}`);
      showTableWithAnimation();

      // Отправляем сообщение в главный процесс
      window.electron.sendToMain('folder-selected', folderPath);
    });

    inputElement.click();
  });

  ipcRenderer.on('update-csv-data', (data) => {
    console.log('Received data from main process:', data);
    // Вызываем функцию для обновления данных в таблице
    updateTableWithData(data);
  });


  function updateTableWithData(data) {
    const tableBody = document.querySelector('tbody');
    // Здесь вы можете выполнить необходимую обработку данных и добавить их в таблицу
    // Например, разбить строку данных на ячейки и добавить их в таблицу
    const dataRows = data.split('\n'); // Разбить данные по строкам

    // Очистить существующие строки в таблице
    tableBody.innerHTML = '';

    // Добавить каждую строку данных в таблицу
    dataRows.forEach((rowData) => {
      const row = document.createElement('tr');
      const dataCells = rowData.split(','); // Разбить строку данных на ячейки

      dataCells.forEach((cellData) => {
        const cell = document.createElement('td');
        cell.textContent = cellData;
        row.appendChild(cell);
      });

      tableBody.appendChild(row);
    });
  }

  function updateResult(resultText) {
    const resultElement = document.getElementById('result-container');
    const resultTextElement = document.getElementById('result-text');
    resultTextElement.textContent = `${resultText}`;

    resultElement.style.opacity = 1;
  }

  function showTableWithAnimation() {
    const dataTable = document.getElementById('data-table');
    dataTable.style.display = 'block';
    setTimeout(() => {
      dataTable.style.opacity = 1;
    }, 100);

    const button = document.getElementById('myButton');
    button.style.opacity = 0;
    button.style.pointerEvents = 'none';
  }
});
