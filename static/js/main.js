// JavaScript for handling file upload, WebSocket connection, and UI updates

const fileInput = document.getElementById('file-input');
const fileDropArea = document.getElementById('file-drop-area');
const fileInfo = document.getElementById('file-info');
const fileNameDisplay = document.getElementById('file-name');
const fileSizeDisplay = document.getElementById('file-size');
const transcribeButton = document.getElementById('transcribe-btn');
const progressSection = document.getElementById('progress-section');
const progressTitle = document.getElementById('progress-title');
const progressBar = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');
const progressMessage = document.getElementById('progress-message');
const progressPercentage = document.getElementById('progress-percentage');
const resultsSection = document.getElementById('results-section');
const transcriptionResult = document.getElementById('transcription-result');
const wordCountDisplay = document.getElementById('word-count');
const detectedLanguage = document.getElementById('detected-language');
const modelUsedDisplay = document.getElementById('model-used');
const timestampDisplay = document.getElementById('timestamp');
const copyButton = document.getElementById('copy-btn');
const downloadButton = document.getElementById('download-btn');
const newFileButton = document.getElementById('new-file-btn');
const errorSection = document.getElementById('error-section');
const errorText = document.getElementById('error-text');
const retryButton = document.getElementById('retry-btn');

// WebSocket connection for real-time progress updates
let websocket;

// Add event listeners
fileInput.addEventListener('change', handleFileSelect);
fileDropArea.addEventListener('click', () => fileInput.click());
fileDropArea.addEventListener('dragover', handleDragOver);
fileDropArea.addEventListener('dragleave', handleDragLeave);
fileDropArea.addEventListener('drop', handleFileDrop);
transcribeButton.addEventListener('click', startTranscription);
copyButton.addEventListener('click', copyToClipboard);
downloadButton.addEventListener('click', downloadTranscription);
newFileButton.addEventListener('click', resetUI);
retryButton.addEventListener('click', retryUpload);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        displayFileInfo(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    fileDropArea.classList.add('dragover');
}

function handleDragLeave(event) {
    fileDropArea.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    fileDropArea.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) {
        displayFileInfo(file);
    }
}

function displayFileInfo(file) {
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    transcribeButton.disabled = false;
    fileInfo.style.display = 'flex';
}

function startTranscription() {
    const file = fileInput.files[0];
    if (!file) return;

    // Reset UI
    errorSection.style.display = 'none';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    // Initialize WebSocket for progress updates
    const clientId = generateUniqueId();
    websocket = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);
    websocket.onmessage = updateProgress;
    websocket.onclose = () => console.log('WebSocket connection closed');

    // Upload file to server
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            fetch(`/transcribe/${data.file_id}`, {
                method: 'POST'
            })
                .then(response => response.json())
                .then(handleTranscriptionResult)
                .catch(showError);
        } else {
            showError({ detail: data.message });
        }
    })
    .catch(showError);
}

function updateProgress(event) {
    const data = JSON.parse(event.data);
    progressTitle.textContent = data.status;
    progressMessage.textContent = data.message;

    const percentage = data.progress || 0;
    progressPercentage.textContent = `${percentage}%`;
    progressFill.style.width = `${percentage}%`;

    if (data.status === 'completed') {
        websocket.close();
    }
}

function handleTranscriptionResult(data) {
    if (!data.success) {
        showError({ detail: data.message });
        return;
    }

    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    transcriptionResult.value = data.transcription.text;
    wordCountDisplay.textContent = data.transcription.word_count;
    detectedLanguage.textContent = data.transcription.language;
    modelUsedDisplay.textContent = data.transcription.model_used;
    timestampDisplay.textContent = new Date(data.transcription.timestamp).toLocaleString();
}

function copyToClipboard() {
    transcriptionResult.select();
    document.execCommand('copy');
    showToast('Transcription copied to clipboard', 'success');
}

function downloadTranscription() {
    const blob = new Blob([transcriptionResult.value], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileNameDisplay.textContent}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

function showError(error) {
    progressSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorText.textContent = error.detail || 'An unexpected error occurred';
}

function showToast(message, type) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.getElementById('toast-container').appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

function retryUpload() {
    errorSection.style.display = 'none';
}

function resetUI() {
    fileInput.value = '';
    fileInfo.style.display = 'none';
    resultsSection.style.display = 'none';
    progressSection.style.display = 'none';
    errorSection.style.display = 'none';
    transcribeButton.disabled = true;
    transcriptionResult.value = '';
}

function generateUniqueId() {
    return `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Handle form submission (prevent default behavior)
const uploadForm = document.getElementById('upload-form');
uploadForm.addEventListener('submit', function(event) {
    event.preventDefault();
    startTranscription();
});

