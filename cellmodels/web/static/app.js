// Global Variables
let selectedImageFile = null;
let apiResponseData = null; // Holds the last prediction response
let overlayVisible = true;
let trainingPollInterval = null;
let telemetryChart = null;
let overlayZoomPan = null;
let heatmapZoomPan = null;

// Zoom and Pan helper class for interactive visualization
class ZoomPan {
    constructor(wrapper, target) {
        this.wrapper = wrapper;
        this.target = target;
        
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        
        this.isDragging = false;
        this.startX = 0;
        this.startY = 0;
        
        this.initEvents();
    }
    
    initEvents() {
        // Prevent default zoom/scroll on mouse wheel over the wrapper
        this.wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            // Zoom direction: positive for scroll up, negative for scroll down
            const direction = e.deltaY < 0 ? 1 : -1;
            this.zoom(direction, e);
        }, { passive: false });
        
        // Drag panning
        this.wrapper.addEventListener('mousedown', (e) => {
            // Drag only on target or wrapper, but ignore if clicking buttons
            if (e.target.closest('.zoom-btn')) return;
            if (e.button !== 0) return; // Left click only
            
            this.isDragging = true;
            this.startX = e.clientX - this.panX;
            this.startY = e.clientY - this.panY;
            this.wrapper.style.cursor = 'grabbing';
            e.preventDefault();
        });
        
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            this.panX = e.clientX - this.startX;
            this.panY = e.clientY - this.startY;
            this.applyTransform();
        });
        
        window.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.wrapper.style.cursor = 'grab';
            }
        });
        
        // Touch support (pinch to zoom and drag)
        let lastTouchDistance = 0;
        this.wrapper.addEventListener('touchstart', (e) => {
            if (e.target.closest('.zoom-btn')) return;
            if (e.touches.length === 1) {
                this.isDragging = true;
                this.startX = e.touches[0].clientX - this.panX;
                this.startY = e.touches[0].clientY - this.panY;
            } else if (e.touches.length === 2) {
                this.isDragging = false;
                lastTouchDistance = this.getTouchDistance(e);
            }
        });
        
        this.wrapper.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1 && this.isDragging) {
                this.panX = e.touches[0].clientX - this.startX;
                this.panY = e.touches[0].clientY - this.startY;
                this.applyTransform();
                e.preventDefault();
            } else if (e.touches.length === 2) {
                e.preventDefault();
                const dist = this.getTouchDistance(e);
                const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                const factor = dist / lastTouchDistance;
                const direction = factor > 1 ? 1 : -1;
                if (Math.abs(factor - 1) > 0.01) {
                    this.zoom(direction, { clientX: midX, clientY: midY });
                    lastTouchDistance = dist;
                }
            }
        }, { passive: false });
        
        this.wrapper.addEventListener('touchend', () => {
            this.isDragging = false;
        });
    }
    
    getTouchDistance(e) {
        return Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
        );
    }
    
    zoom(direction, mouseXOrEvent, mouseY) {
        const oldScale = this.scale;
        const zoomStep = 1.15;
        
        if (direction > 0) {
            this.scale = Math.min(this.scale * zoomStep, 8.0);
        } else {
            this.scale = Math.max(this.scale / zoomStep, 0.25);
        }
        
        let clientX, clientY;
        if (mouseXOrEvent instanceof MouseEvent || mouseXOrEvent instanceof TouchEvent || (mouseXOrEvent && mouseXOrEvent.clientX !== undefined)) {
            if (mouseXOrEvent.target && mouseXOrEvent.target.closest('.zoom-btn')) {
                clientX = undefined;
                clientY = undefined;
            } else {
                clientX = mouseXOrEvent.clientX;
                clientY = mouseXOrEvent.clientY;
            }
        } else {
            clientX = mouseXOrEvent;
            clientY = mouseY;
        }
        
        if (clientX !== undefined && clientY !== undefined) {
            const rect = this.wrapper.getBoundingClientRect();
            const mx = clientX - rect.left;
            const my = clientY - rect.top;
            
            const tx = (mx - this.panX) / oldScale;
            const ty = (my - this.panY) / oldScale;
            
            this.panX = mx - tx * this.scale;
            this.panY = my - ty * this.scale;
        } else {
            const rect = this.wrapper.getBoundingClientRect();
            const mx = rect.width / 2;
            const my = rect.height / 2;
            
            const tx = (mx - this.panX) / oldScale;
            const ty = (my - this.panY) / oldScale;
            
            this.panX = mx - tx * this.scale;
            this.panY = my - ty * this.scale;
        }
        
        this.applyTransform();
    }
    
    reset() {
        const wRect = this.wrapper.getBoundingClientRect();
        
        let targetWidth = 0;
        let targetHeight = 0;
        
        if (this.target.tagName === 'CANVAS') {
            targetWidth = this.target.width;
            targetHeight = this.target.height;
        } else if (this.target.tagName === 'IMG') {
            targetWidth = this.target.naturalWidth;
            targetHeight = this.target.naturalHeight;
        } else {
            const canvas = this.target.querySelector('canvas');
            if (canvas) {
                targetWidth = canvas.width;
                targetHeight = canvas.height;
            } else {
                const img = this.target.querySelector('img');
                if (img) {
                    targetWidth = img.naturalWidth;
                    targetHeight = img.naturalHeight;
                } else {
                    targetWidth = this.target.offsetWidth;
                    targetHeight = this.target.offsetHeight;
                }
            }
        }
        
        if (!targetWidth || !targetHeight) {
            this.scale = 1;
            this.panX = 0;
            this.panY = 0;
            this.applyTransform();
            return;
        }
        
        const padding = 16;
        const availWidth = Math.max(100, wRect.width - padding * 2);
        const availHeight = Math.max(100, wRect.height - padding * 2);
        
        const scaleX = availWidth / targetWidth;
        const scaleY = availHeight / targetHeight;
        this.scale = Math.min(scaleX, scaleY);
        
        const scaledWidth = targetWidth * this.scale;
        const scaledHeight = targetHeight * this.scale;
        
        this.panX = (wRect.width - scaledWidth) / 2;
        this.panY = (wRect.height - scaledHeight) / 2;
        
        this.target.style.width = targetWidth + 'px';
        this.target.style.height = targetHeight + 'px';
        
        this.applyTransform();
        this.wrapper.style.cursor = 'grab';
    }
    
    applyTransform() {
        this.target.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.scale})`;
    }
}

// Initial Setup
document.addEventListener('DOMContentLoaded', () => {
    // Setup drag and drop for image upload
    const dropzone = document.getElementById('dropzone');
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    }, false);

    // Initialize zoom pan instances
    const overlayWrapper = document.getElementById('overlay-wrapper');
    const overlayContainer = document.getElementById('overlay-container');
    overlayZoomPan = new ZoomPan(overlayWrapper, overlayContainer);
    
    const heatmapWrapper = document.getElementById('heatmap-wrapper');
    const heatmapContainer = document.getElementById('heatmap-container');
    heatmapZoomPan = new ZoomPan(heatmapWrapper, heatmapContainer);

    // Initialize telemetry chart
    initChart();

    // Load initial calibrated parameters for default magnification
    handleMagnificationChange();
});

// Window Resize Handler to update zoom fit
window.addEventListener('resize', () => {
    if (overlayZoomPan && document.getElementById('overlay-container').style.display !== 'none') {
        overlayZoomPan.reset();
    }
    if (heatmapZoomPan && document.getElementById('heatmap-container').style.display !== 'none') {
        heatmapZoomPan.reset();
    }
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(`tab-${tabId}`).classList.add('active');
    document.getElementById(`tab-${tabId}-btn`).classList.add('active');
}

// Prediction sandbox tab switching
function switchOutputTab(tabId) {
    document.querySelectorAll('.visual-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.output-tab').forEach(tab => {
        tab.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');
    // Find output-tab button with matching click attribute
    const tabBtns = document.querySelectorAll('.output-tab');
    tabBtns.forEach(btn => {
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });

    // Reset zoom when switching tabs to ensure layouts are correctly sized
    if (tabId === 'tab-overlay' && overlayZoomPan) {
        setTimeout(() => overlayZoomPan.reset(), 50);
    } else if (tabId === 'tab-heatmap' && heatmapZoomPan) {
        setTimeout(() => heatmapZoomPan.reset(), 50);
    }
}

// Image Selection Handling
function handleImageSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, TIF, JPG).');
        return;
    }
    selectedImageFile = file;
    
    // Update UI
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('image-preview-container').style.display = 'flex';
        document.getElementById('dropzone').style.display = 'none';
        
        // Load on canvas
        const img = new Image();
        img.onload = () => {
            const baseCanvas = document.getElementById('base-canvas');
            const overlayCanvas = document.getElementById('overlay-canvas');
            const ctx = baseCanvas.getContext('2d');
            
            // Set base dimensions
            baseCanvas.width = img.width;
            baseCanvas.height = img.height;
            overlayCanvas.width = img.width;
            overlayCanvas.height = img.height;
            
            ctx.drawImage(img, 0, 0);
            
            // Show canvas container, hide placeholder
            document.getElementById('overlay-container').style.display = 'grid';
            document.getElementById('overlay-placeholder').style.display = 'none';
            document.getElementById('overlay-zoom-controls').style.display = 'flex';
            
            // Reset overlay zoom/pan
            if (overlayZoomPan) {
                setTimeout(() => overlayZoomPan.reset(), 50);
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function clearSelectedImage(event) {
    event.stopPropagation();
    selectedImageFile = null;
    apiResponseData = null;
    
    document.getElementById('image-upload').value = '';
    document.getElementById('image-preview-container').style.display = 'none';
    document.getElementById('dropzone').style.display = 'flex';
    document.getElementById('overlay-container').style.display = 'none';
    document.getElementById('overlay-placeholder').style.display = 'block';
    document.getElementById('overlay-controls').style.display = 'none';
    document.getElementById('overlay-zoom-controls').style.display = 'none';
    document.getElementById('confluency-pct-val').innerText = '—';
    
    // Clear Heatmap tab
    document.getElementById('heatmap-placeholder').style.display = 'block';
    document.getElementById('heatmap-zoom-controls').style.display = 'none';
    const heatmapContainer = document.getElementById('heatmap-container');
    heatmapContainer.style.display = 'none';
    const heatmapImg = document.getElementById('heatmap-img');
    heatmapImg.src = '';
}

// UI Configuration Controls Toggles
function toggleCheckpointUpload() {
    const mag = document.getElementById('mag-select').value;
    const checkpointGroup = document.getElementById('checkpoint-upload-group');
    if (mag === 'custom') {
        checkpointGroup.style.display = 'flex';
    } else {
        checkpointGroup.style.display = 'none';
    }
}

async function fetchCalibratedParameters(magnification) {
    try {
        const response = await fetch(`/api/parameters/${magnification}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch parameters for ${magnification}`);
        }
        const data = await response.json();
        
        // Update Threshold Method checkbox
        const useAbsoluteCheckbox = document.getElementById('use-absolute-threshold');
        const isAbsolute = (data.method === 'absolute_threshold');
        useAbsoluteCheckbox.checked = isAbsolute;
        toggleThresholdMethod(); // updates container displays
        
        // Update sliders values and their display labels
        if (isAbsolute) {
            if (data.prob_threshold !== undefined) {
                document.getElementById('prob-threshold').value = data.prob_threshold;
                updateSliderValue('prob-threshold');
            }
        } else {
            const factor = data.t_factor !== undefined ? data.t_factor : data.threshold_factor;
            if (factor !== undefined) {
                document.getElementById('otsu-factor').value = factor;
                updateSliderValue('otsu-factor');
            }
        }
        
        if (data.closing_radius !== undefined) {
            document.getElementById('closing-radius').value = data.closing_radius;
            updateSliderValue('closing-radius');
        }
        
        if (data.min_object_size !== undefined) {
            document.getElementById('min-object-size').value = data.min_object_size;
            updateSliderValue('min-object-size');
        }
        
    } catch (e) {
        console.error('Error fetching calibrated parameters:', e);
    }
}

async function handleMagnificationChange() {
    toggleCheckpointUpload();
    const mag = document.getElementById('mag-select').value;
    if (mag !== 'custom') {
        await fetchCalibratedParameters(mag);
    }
}

function toggleThresholdMethod() {
    const isAbsolute = document.getElementById('use-absolute-threshold').checked;
    const otsuContainer = document.getElementById('otsu-container');
    const absoluteContainer = document.getElementById('absolute-container');
    
    if (isAbsolute) {
        otsuContainer.style.display = 'none';
        absoluteContainer.style.display = 'flex';
    } else {
        otsuContainer.style.display = 'flex';
        absoluteContainer.style.display = 'none';
    }
}

function updateSliderValue(sliderId) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(`${sliderId}-val`).innerText = val;
}

// Submitting Prediction
async function submitPrediction() {
    if (!selectedImageFile) {
        alert('Please upload a micrograph image first.');
        return;
    }

    const btn = document.getElementById('run-prediction-btn');
    btn.disabled = true;
    btn.innerText = 'Analyzing...';

    const formData = new FormData();
    formData.append('file', selectedImageFile);
    
    const mag = document.getElementById('mag-select').value;
    formData.append('magnification', mag);

    if (mag === 'custom') {
        const cpFile = document.getElementById('checkpoint-upload').files[0];
        if (cpFile) {
            formData.append('checkpoint_file', cpFile);
        }
    }

    // Parameters
    const closingRadius = document.getElementById('closing-radius').value;
    const minObjectSize = document.getElementById('min-object-size').value;
    formData.append('closing_radius', closingRadius);
    formData.append('min_object_size', minObjectSize);

    const useAbsolute = document.getElementById('use-absolute-threshold').checked;
    if (useAbsolute) {
        const probThreshold = document.getElementById('prob-threshold').value;
        formData.append('prob_threshold', probThreshold);
    } else {
        const tFactor = document.getElementById('otsu-factor').value;
        formData.append('threshold_factor', tFactor);
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed.');
        }

        apiResponseData = await response.json();
        
        // Update Confluency metrics
        document.getElementById('confluency-pct-val').innerText = `${apiResponseData.confluency_pct}%`;
        
        // Show Heatmap
        const heatmapImg = document.getElementById('heatmap-img');
        const heatmapContainer = document.getElementById('heatmap-container');
        heatmapImg.src = apiResponseData.density_map;
        
        heatmapImg.onload = () => {
            if (heatmapZoomPan) {
                setTimeout(() => heatmapZoomPan.reset(), 50);
            }
        };
        
        heatmapContainer.style.display = 'block';
        document.getElementById('heatmap-placeholder').style.display = 'none';
        document.getElementById('heatmap-zoom-controls').style.display = 'flex';

        // Redraw canvases with masks and borders
        overlayVisible = true;
        renderMaskOverlay();
        
        document.getElementById('overlay-controls').style.display = 'flex';
        
        if (overlayZoomPan) {
            setTimeout(() => overlayZoomPan.reset(), 50);
        }
        
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Analyze Micrograph';
    }
}

// Overlay Render Logic (Green fill, red outline boundary)
function renderMaskOverlay() {
    if (!apiResponseData || !apiResponseData.cell_mask) return;

    const baseCanvas = document.getElementById('base-canvas');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const ctx = overlayCanvas.getContext('2d');
    const opacity = document.getElementById('overlay-opacity').value;

    if (!overlayVisible) {
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        return;
    }

    const maskImg = new Image();
    maskImg.onload = () => {
        const width = maskImg.width;
        const height = maskImg.height;
        
        // Keep canvas dims aligned
        overlayCanvas.width = width;
        overlayCanvas.height = height;

        // Draw offscreen mask for reading pixels
        const offscreen = document.createElement('canvas');
        offscreen.width = width;
        offscreen.height = height;
        const offCtx = offscreen.getContext('2d');
        offCtx.drawImage(maskImg, 0, 0);
        
        const imgData = offCtx.getImageData(0, 0, width, height);
        const data = imgData.data;

        // Generate output pixels
        const outData = ctx.createImageData(width, height);
        const out = outData.data;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const isActive = data[idx] > 128; // white cell body

                if (isActive) {
                    let isBorder = false;
                    if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
                        isBorder = true;
                    } else {
                        // 4-connectivity border scan
                        const top = idx - width * 4;
                        const bottom = idx + width * 4;
                        const left = idx - 4;
                        const right = idx + 4;
                        
                        if (data[top] <= 128 || data[bottom] <= 128 || data[left] <= 128 || data[right] <= 128) {
                            isBorder = true;
                        }
                    }

                    if (isBorder) {
                        out[idx] = 239;     // R
                        out[idx + 1] = 68;  // G
                        out[idx + 2] = 68;  // B
                        out[idx + 3] = 255; // Alpha
                    } else {
                        out[idx] = 16;      // R
                        out[idx + 1] = 185; // G
                        out[idx + 2] = 129; // B
                        out[idx + 3] = Math.round(opacity * 255); // Alpha
                    }
                } else {
                    out[idx + 3] = 0; // Transparent
                }
            }
        }
        ctx.putImageData(outData, 0, 0);
    };
    maskImg.src = apiResponseData.cell_mask;
}

function adjustOverlayOpacity() {
    renderMaskOverlay();
}

function toggleOverlayLayer() {
    overlayVisible = !overlayVisible;
    renderMaskOverlay();
}

// -------------------------------------------------------------
// TRAINING OPERATIONS
// -------------------------------------------------------------

function initChart() {
    const canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.05)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.02)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.2
                },
                {
                    label: 'Val Dice',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#9ca3af'
                    },
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#9ca3af'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    },
                    title: {
                        display: true,
                        text: 'Dice Coefficient',
                        color: '#9ca3af'
                    },
                    min: 0,
                    max: 1
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#9ca3af'
                    },
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#9ca3af'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f3f4f6'
                    }
                }
            }
        }
    });
}

async function startTraining() {
    const trainDir = document.getElementById('train-dir-input').value.trim();
    const valDir = document.getElementById('val-dir-input').value.trim();
    const outputDir = document.getElementById('output-dir-input').value.trim();
    const epochs = parseInt(document.getElementById('train-epochs').value);
    const batchSize = parseInt(document.getElementById('train-batch-size').value);
    const lr = parseFloat(document.getElementById('train-lr').value);
    const seed = parseInt(document.getElementById('train-seed').value);
    const backbone = document.getElementById('train-backbone').value;

    if (!trainDir || !valDir) {
        alert('Train and Validation directory paths are required.');
        return;
    }

    const startBtn = document.getElementById('start-train-btn');
    const stopBtn = document.getElementById('stop-train-btn');
    
    startBtn.disabled = true;
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-flex';

    // Clear chart
    telemetryChart.data.labels = [];
    telemetryChart.data.datasets[0].data = [];
    telemetryChart.data.datasets[1].data = [];
    telemetryChart.data.datasets[2].data = [];
    telemetryChart.update();

    const formData = new FormData();
    formData.append('train_dir', trainDir);
    formData.append('val_dir', valDir);
    formData.append('output_dir', outputDir);
    formData.append('epochs', epochs);
    formData.append('batch_size', batchSize);
    formData.append('lr', lr);
    formData.append('seed', seed);
    formData.append('encoder_backbone', backbone);

    try {
        const response = await fetch('/api/train/start', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start training run.');
        }

        // Start polling status
        document.getElementById('train-status-label').innerText = 'INITIALIZING';
        document.getElementById('console-logs').innerText = 'Initializing training loop...';
        
        trainingPollInterval = setInterval(pollTrainingStatus, 1000);
        
    } catch (e) {
        alert(`Error: ${e.message}`);
        startBtn.disabled = false;
        startBtn.style.display = 'inline-flex';
        stopBtn.style.display = 'none';
    }
}

async function stopTraining() {
    if (!confirm('Are you sure you want to stop the training run? Checkpoints saved so far will remain.')) {
        return;
    }

    try {
        await fetch('/api/train/stop', { method: 'POST' });
        document.getElementById('train-status-label').innerText = 'ABORTING';
    } catch (e) {
        console.error('Error stopping training:', e);
    }
}

async function pollTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        const data = await response.json();

        // Update progress UI
        const progressPct = Math.round(data.progress * 100);
        document.getElementById('train-progress-pct').innerText = `${progressPct}%`;
        
        if (data.is_running) {
            document.getElementById('train-status-label').innerText = 'TRAINING';
            document.getElementById('train-epoch-label').innerText = `Epoch ${data.current_epoch} / ${data.total_epochs}`;
        } else {
            document.getElementById('train-status-label').innerText = 'IDLE';
            document.getElementById('train-epoch-label').innerText = `Epoch ${data.current_epoch} / ${data.total_epochs}`;
            
            // Clean up polling
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
            
            document.getElementById('start-train-btn').disabled = false;
            document.getElementById('start-train-btn').style.display = 'inline-flex';
            document.getElementById('stop-train-btn').style.display = 'none';
        }

        // Render console logs
        const logBox = document.getElementById('console-logs');
        logBox.innerText = data.logs.join('\n');
        logBox.scrollTop = logBox.scrollHeight;

        // Render metrics cards
        if (data.metrics.length > 0) {
            const latest = data.metrics[data.metrics.length - 1];
            document.getElementById('metric-train-loss').innerText = latest.train_loss.toFixed(4);
            document.getElementById('metric-val-loss').innerText = latest.val_loss.toFixed(4);
            document.getElementById('metric-val-dice').innerText = latest.val_dice.toFixed(4);
            
            // Update chart data
            const labels = data.metrics.map(m => `Epoch ${m.epoch}`);
            const trainLosses = data.metrics.map(m => m.train_loss);
            const valLosses = data.metrics.map(m => m.val_loss);
            const valDices = data.metrics.map(m => m.val_dice);

            telemetryChart.data.labels = labels;
            telemetryChart.data.datasets[0].data = trainLosses;
            telemetryChart.data.datasets[1].data = valLosses;
            telemetryChart.data.datasets[2].data = valDices;
            telemetryChart.update();
        }

    } catch (e) {
        console.error('Error polling status:', e);
    }
}
