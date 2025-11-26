// Global variables
let personFile = null;
let clothFile = null;
let personName = '';
let clothName = '';
let processingInterval = null;

// DOM Elements
const personInput = document.getElementById('personInput');
const clothInput = document.getElementById('clothInput');
const personUploadArea = document.getElementById('personUploadArea');
const clothUploadArea = document.getElementById('clothUploadArea');
const personPreview = document.getElementById('personPreview');
const clothPreview = document.getElementById('clothPreview');
const startBtn = document.getElementById('startBtn');
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressPercent = document.getElementById('progressPercent');
const statusText = document.getElementById('statusText');
const downloadBtn = document.getElementById('downloadBtn');
const tryAgainBtn = document.getElementById('tryAgainBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Person image upload
    personUploadArea.addEventListener('click', () => personInput.click());
    personInput.addEventListener('change', (e) => handleFileSelect(e, 'person'));
    
    // Cloth image upload
    clothUploadArea.addEventListener('click', () => clothInput.click());
    clothInput.addEventListener('change', (e) => handleFileSelect(e, 'cloth'));
    
    // Drag and drop
    setupDragDrop(personUploadArea, personInput, 'person');
    setupDragDrop(clothUploadArea, clothInput, 'cloth');
    
    // Start button
    startBtn.addEventListener('click', startProcessing);
    
    // Result actions
    downloadBtn.addEventListener('click', downloadResult);
    tryAgainBtn.addEventListener('click', resetApp);
}

function setupDragDrop(area, input, type) {
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });
    
    area.addEventListener('dragleave', () => {
        area.classList.remove('dragover');
    });
    
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            input.files = files;
            handleFileSelect({ target: input }, type);
        }
    });
}

function handleFileSelect(event, type) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.match('image/jpeg') && !file.type.match('image/jpg')) {
        alert('Please select a JPG or JPEG image only.');
        return;
    }
    
    // Store file
    if (type === 'person') {
        personFile = file;
        displayPreview(file, personPreview, personUploadArea);
    } else {
        clothFile = file;
        displayPreview(file, clothPreview, clothUploadArea);
    }
    
    // Enable start button if both files are selected
    updateStartButton();
}

function displayPreview(file, previewElement, uploadArea) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewElement.src = e.target.result;
        previewElement.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

function updateStartButton() {
    if (personFile && clothFile) {
        startBtn.disabled = false;
        startBtn.style.opacity = '1';
    } else {
        startBtn.disabled = true;
        startBtn.style.opacity = '0.5';
    }
}

async function startProcessing() {
    // Disable button
    startBtn.disabled = true;
    
    try {
        // Upload files
        const uploadSuccess = await uploadFiles();
        
        if (!uploadSuccess) {
            alert('Failed to upload files. Please try again.');
            startBtn.disabled = false;
            return;
        }
        
        // Switch to processing view
        uploadSection.style.display = 'none';
        processingSection.style.display = 'block';
        processingSection.classList.add('fade-in');
        
        // Set preview images
        document.getElementById('procPersonImg').src = personPreview.src;
        document.getElementById('procClothImg').src = clothPreview.src;
        
        // Start processing
        const processSuccess = await startVirtualTryOn();
        
        if (!processSuccess) {
            alert('Failed to start processing. Please try again.');
            resetApp();
            return;
        }
        
        // Start polling for progress
        startProgressPolling();
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        resetApp();
    }
}

async function uploadFiles() {
    const formData = new FormData();
    formData.append('person_image', personFile);
    formData.append('cloth_image', clothFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            personName = data.person_name;
            clothName = data.cloth_name;
            return true;
        } else {
            console.error('Upload failed:', data.error);
            return false;
        }
    } catch (error) {
        console.error('Upload error:', error);
        return false;
    }
}

async function startVirtualTryOn() {
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                person_name: personName,
                cloth_name: clothName
            })
        });
        
        const data = await response.json();
        return data.success;
    } catch (error) {
        console.error('Process start error:', error);
        return false;
    }
}

function startProgressPolling() {
    processingInterval = setInterval(async () => {
        try {
            const response = await fetch('/progress');
            const data = await response.json();
            
            updateProgress(data);
            
            // Check if complete
            if (data.stage === 'complete') {
                clearInterval(processingInterval);
                setTimeout(() => showResults(), 1000);
            } else if (data.stage === 'error') {
                clearInterval(processingInterval);
                alert('Error: ' + data.message);
                resetApp();
            }
        } catch (error) {
            console.error('Progress polling error:', error);
        }
    }, 1000); // Poll every second
}

function updateProgress(data) {
    const progress = data.progress || 0;
    const message = data.message || 'Processing...';
    
    // Update progress bar
    progressFill.style.width = progress + '%';
    progressPercent.textContent = progress + '%';
    
    // Update stage text
    let stageText = '';
    switch (data.stage) {
        case 'starting':
            stageText = 'Initializing';
            break;
        case 'preprocessing':
            stageText = 'Preprocessing';
            break;
        case 'inference':
            stageText = 'AI Processing';
            break;
        case 'complete':
            stageText = 'Complete';
            break;
        default:
            stageText = 'Processing';
    }
    
    progressText.textContent = stageText;
    statusText.textContent = message;
    
    // Update status icon based on stage
    const statusIcon = document.querySelector('.status-icon');
    if (data.stage === 'preprocessing') {
        statusIcon.textContent = '🔍';
    } else if (data.stage === 'inference') {
        statusIcon.textContent = '🎨';
    } else if (data.stage === 'complete') {
        statusIcon.textContent = '✅';
    }
}

function showResults() {
    // Hide processing, show results
    processingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Set result images
    const resultFilename = `${personName}_${clothName}.jpg`;
    
    document.getElementById('finalPersonImg').src = `/uploads/person/${personName}.jpg`;
    document.getElementById('finalClothImg').src = `/uploads/cloth/${clothName}.jpg`;
    document.getElementById('finalResultImg').src = `/output/${resultFilename}`;
    
    // Also update processing section result
    document.getElementById('procResultImg').src = `/output/${resultFilename}`;
    document.getElementById('procResultImg').style.display = 'block';
    document.getElementById('resultPlaceholder').style.display = 'none';
}

function downloadResult() {
    const resultFilename = `${personName}_${clothName}.jpg`;
    const link = document.createElement('a');
    link.href = `/output/${resultFilename}`;
    link.download = `virtual_tryon_${Date.now()}.jpg`;
    link.click();
}

function resetApp() {
    // Clear interval
    if (processingInterval) {
        clearInterval(processingInterval);
    }
    
    // Reset variables
    personFile = null;
    clothFile = null;
    personName = '';
    clothName = '';
    
    // Reset file inputs
    personInput.value = '';
    clothInput.value = '';
    
    // Reset previews
    personPreview.style.display = 'none';
    clothPreview.style.display = 'none';
    personUploadArea.querySelector('.upload-placeholder').style.display = 'block';
    clothUploadArea.querySelector('.upload-placeholder').style.display = 'block';
    
    // Clear output images
    document.getElementById('procResultImg').src = '';
    document.getElementById('procResultImg').style.display = 'none';
    document.getElementById('resultPlaceholder').style.display = 'flex';
    document.getElementById('finalPersonImg').src = '';
    document.getElementById('finalClothImg').src = '';
    document.getElementById('finalResultImg').src = '';
    
    // Reset progress
    progressFill.style.width = '0%';
    progressPercent.textContent = '0%';
    progressText.textContent = 'Starting...';
    statusText.textContent = 'Initializing virtual try-on process...';
    
    // Show upload section
    uploadSection.style.display = 'block';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Reset button
    startBtn.disabled = true;
    
    // Scroll to top
    window.scrollTo(0, 0);
}

// Utility: Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Error handling
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

// Prevent page from navigating on drag/drop
window.addEventListener('dragover', (e) => {
    e.preventDefault();
}, false);

window.addEventListener('drop', (e) => {
    e.preventDefault();
}, false);

// Image Modal Functions
function openImageModal(imageSrc) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    
    modal.style.display = 'flex';
    modalImg.src = imageSrc;
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
    
    // Re-enable body scroll
    document.body.style.overflow = 'auto';
}

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeImageModal();
    }
});
