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
const speakerCountDisplay = document.getElementById('speaker-count');
const timestampDisplay = document.getElementById('timestamp');
const speakerSection = document.getElementById('speaker-section');
const speakerTimeline = document.getElementById('speaker-timeline');
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

    // NOTE: Backend generates client_id internally; progress via WS may be limited
    // We still open a generic socket to surface server-side broadcast if configured.
    try {
        const clientId = generateUniqueId();
        websocket = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);
        websocket.onmessage = updateProgress;
        websocket.onclose = () => console.log('WebSocket connection closed');
    } catch (e) {
        console.warn('WebSocket unavailable', e);
    }

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
            // Gather options
            const language = (document.getElementById('language-select')?.value || 'hi');
            const accuracy_mode = !!document.getElementById('accuracy-toggle')?.checked;
            const prompt = (document.getElementById('prompt-text')?.value || '').trim();
            const domainTermsRaw = (document.getElementById('domain-terms')?.value || '').trim();
            const domain_terms = domainTermsRaw ? domainTermsRaw.split(',').map(s => s.trim()).filter(Boolean) : [];

            fetch(`/transcribe/${data.file_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language, accuracy_mode, prompt, domain_terms })
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
    detectedLanguage.textContent = data.transcription.language_name || data.transcription.language;
    modelUsedDisplay.textContent = data.transcription.model_used;
    timestampDisplay.textContent = new Date(data.transcription.timestamp).toLocaleString();
    
    // Handle speaker diarization results
    if (data.transcription.speaker_diarization && data.transcription.speaker_diarization.enabled) {
        const speakerData = data.transcription.speaker_diarization;
        speakerCountDisplay.textContent = speakerData.speaker_count || 0;
        
        if (speakerData.segments && speakerData.segments.length > 0) {
            displaySpeakerTimeline(speakerData.segments);
            speakerSection.style.display = 'block';
        } else {
            speakerSection.style.display = 'none';
        }
    } else {
        speakerCountDisplay.textContent = 'N/A';
        speakerSection.style.display = 'none';
    }
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

function displaySpeakerTimeline(segments) {
    speakerTimeline.innerHTML = '';
    
    if (!segments || segments.length === 0) {
        speakerTimeline.innerHTML = '<p>No speaker segments detected</p>';
        return;
    }
    
    // Group segments by speaker
    const speakers = {};
    segments.forEach(segment => {
        if (!speakers[segment.speaker]) {
            speakers[segment.speaker] = [];
        }
        speakers[segment.speaker].push(segment);
    });
    
    // Create speaker timeline
    const timelineDiv = document.createElement('div');
    timelineDiv.className = 'speaker-timeline';
    
    Object.keys(speakers).forEach((speaker, index) => {
        const speakerDiv = document.createElement('div');
        speakerDiv.className = 'speaker-info';
        speakerDiv.innerHTML = `
            <div class="speaker-header">
                <span class="speaker-label">Speaker ${speaker}</span>
                <span class="speaker-time">${speakers[speaker].length} segments</span>
            </div>
            <div class="speaker-segments">
                ${speakers[speaker].map(seg => 
                    `<span class="segment" title="${seg.start_time}s - ${seg.end_time}s">
                        ${formatTime(seg.start_time)} - ${formatTime(seg.end_time)}
                     </span>`
                ).join('')}
            </div>
        `;
        timelineDiv.appendChild(speakerDiv);
    });
    
    speakerTimeline.appendChild(timelineDiv);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Form submission is handled by button click, no form element needed
