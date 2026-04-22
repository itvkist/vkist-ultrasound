const API_BASE = 'http://127.0.0.1:8000';
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
let currentResult = null;
let originalImageBase64 = null;

const ANGLE_NAMES = {
    'med-lat': 'Med-Lat Long',
    'post-trans': 'Post Trans',
    'sup-trans-flex': 'Suprapat Trans Flex',
    'sup-up-long': 'Suprapat Up Long'
};

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#3498db';
    uploadArea.style.background = '#f8f9fa';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#bdc3c7';
    uploadArea.style.background = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#bdc3c7';
    uploadArea.style.background = 'transparent';
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh hợp lệ');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageBase64 = e.target.result;
        document.getElementById('originalImage').src = originalImageBase64;
        document.getElementById('originalImageContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);

    uploadAndAnalyze(file);
}

async function uploadAndAnalyze(file) {
    showLoading(true);
    hideError();
    hideResults();

    const angleModelValue = document.getElementById('angleModel').value;
    const inflammationModelValue = document.getElementById('inflammationModel').value;
    const segmentModelSupValue = document.getElementById('segmentModelSup').value;
    const segmentModelPostValue = document.getElementById('segmentModelPost').value;
    
    console.log('🎯 Selected models:', {
        angle: angleModelValue,
        inflammation: inflammationModelValue,
        seg_sup: segmentModelSupValue,
        seg_post: segmentModelPostValue
    });

    const formData = new FormData();
    formData.append('image', file);

    try {
        const url = `${API_BASE}/api/analyze?angle_model=${angleModelValue}&inflammation_model=${inflammationModelValue}&segment_model_sup=${segmentModelSupValue}&segment_model_post=${segmentModelPostValue}`;
        console.log('🌐 Request URL:', url);
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Lỗi phân tích');

        const result = await response.json();
        console.log('✅ TRẢ VỀ TỪ API:', result);
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
        }
        else showError('Phân tích thất bại');

    } catch (error) {
        console.error('❌ Error:', error);
        showError(`Lỗi: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    try {
        console.log('--- ĐANG HIỂN THỊ KẾT QUẢ ---');
        document.getElementById('noResults').style.display = 'none';
        
        // Cập nhật ảnh gốc thành ảnh đã tăng cường tương phản (CLAHE)
        if (result.images && result.images.enhanced) {
            document.getElementById('originalImage').src = result.images.enhanced;
        }
        
        document.getElementById('angleResultCard').style.display = 'block';
        document.getElementById('angleValue').textContent = ANGLE_NAMES[result.angle.class] || result.angle.class;
        document.getElementById('angleConfidenceText').textContent = `Độ tin cậy: ${result.angle.confidence}%`;

        if (result.inflammation) {
            console.log('✅ Hiển thị Inflammation');
            document.getElementById('resultsGrid').style.display = 'block';
            document.getElementById('inflammationCard').style.display = 'block';
            
            const inflDiv = document.getElementById('inflammationResult');
            if (result.inflammation.detected) {
                inflDiv.innerHTML = '<span class="badge badge-danger">CÓ KHẢ NĂNG VIÊM / THEO DÕI VIÊM</span>';
            } else {
                inflDiv.innerHTML = '<span class="badge badge-success">KHÔNG VIÊM</span>';
            }
            document.getElementById('inflammationConfidence').textContent = `Độ tin cậy: ${result.inflammation.confidence}%`;
        }

        if (result.segmentation && result.segmentation.performed) {
            console.log('✅ Hiển thị Segmentation & Overlay');
            document.getElementById('segmentedImageContainer').style.display = 'block';
            document.getElementById('segmentedImage').src = result.images.segmented;
            
            if (result.segmentation.color_legend) {
                displayColorLegend(result.segmentation.color_legend, result.segmentation.angle_type);
            }

            if (result.segmentation.angle_type === 'sup') {
                document.getElementById('measurementNote').style.display = 'inline';
            } else {
                document.getElementById('measurementNote').style.display = 'none';
            }
        }

        if (result.measurement) {
            console.log('✅ Hiển thị Đo đạc (Measurement)');
            document.getElementById('measurementCard').style.display = 'block';
            
            document.getElementById('thicknessMm').textContent = `${result.measurement.thickness_mm} mm`;
            document.getElementById('thicknessPx').textContent = `${result.measurement.thickness_px}`;
            document.getElementById('measurementLocationX').textContent = result.measurement.location_x;
            document.getElementById('measurementRangeY').textContent = 
                `${result.measurement.y_start} - ${result.measurement.y_end}`;
        }

        if (result.severity) {
            console.log('✅ Hiển thị Mức độ (Severity)');
            document.getElementById('severityCard').style.display = 'block';
            const badge = document.getElementById('severityBadge');
            badge.textContent = result.severity.severity;
            badge.style.background = result.severity.color;
            document.getElementById('severityDescription').textContent = result.severity.description;
            document.getElementById('effusionStat').textContent = 
                `${result.severity.effusion.ratio}% (${result.severity.effusion.thickness}px)`;
            document.getElementById('synoviumStat').textContent = `${result.severity.synovium.ratio}%`;
        }

        // Hiện bảng nhập liệu bệnh nhân
        document.getElementById('patientInfoPanel').style.display = 'block';
        updateSaveButtonState();

        // HIỂN THỊ POPUP KẾT QUẢ
        showResultPopup(result);

    } catch (err) {
        console.error('❌ LỖI TRONG displayResults:', err);
        showError(`Lỗi hiển thị: ${err.message}`);
    }
}

// Kiểm tra tính hợp lệ của form
function updateSaveButtonState() {
    const name = document.getElementById('patientName').value.trim();
    const id = document.getElementById('patientId').value.trim();
    const btn = document.getElementById('saveDataBtn');
    const exportBtn = document.getElementById('exportPdfBtn');
    
    const isValid = !!(currentResult && name && id);
    btn.disabled = !isValid;
    exportBtn.disabled = !isValid;
}

// Lắng nghe thay đổi trên form
['patientName', 'patientId', 'patientGender', 'patientAge', 'doctorDiagnosis'].forEach(id => {
    document.getElementById(id).addEventListener('input', updateSaveButtonState);
});

// Hàm lưu dữ liệu
document.getElementById('saveDataBtn').addEventListener('click', async () => {
    if (!currentResult) return;

    const saveBtn = document.getElementById('saveDataBtn');
    const statusDiv = document.getElementById('saveStatus');
    
    const payload = {
        patient_info: {
            name: document.getElementById('patientName').value,
            id: document.getElementById('patientId').value,
            gender: document.getElementById('patientGender').value,
            age: document.getElementById('patientAge').value,
            diagnosis: document.getElementById('doctorDiagnosis').value
        },
        analysis_result: currentResult,
        images: {
            original: originalImageBase64,
            segmented: currentResult.images.segmented
        }
    };

    try {
        saveBtn.disabled = true;
        statusDiv.innerHTML = '<span style="color: blue;">⌛ Đang lưu...</span>';
        
        const response = await fetch(`${API_BASE}/api/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const resData = await response.json();
        if (resData.success) {
            statusDiv.innerHTML = `<span style="color: green;">✅ Đã lưu vào thư mục: <strong>${resData.folder}</strong></span>`;
        } else {
            throw new Error(resData.detail || 'Lỗi không xác định');
        }
    } catch (error) {
        console.error('❌ Save error:', error);
        statusDiv.innerHTML = `<span style="color: red;">❌ Lỗi: ${error.message}</span>`;
    } finally {
        saveBtn.disabled = false;
    }
});

document.getElementById('exportPdfBtn').addEventListener('click', async () => {
    if (!currentResult) return;

    const exportBtn = document.getElementById('exportPdfBtn');
    const statusDiv = document.getElementById('saveStatus');
    
    const payload = {
        patient_info: {
            name: document.getElementById('patientName').value,
            id: document.getElementById('patientId').value,
            gender: document.getElementById('patientGender').value,
            age: document.getElementById('patientAge').value,
            diagnosis: document.getElementById('doctorDiagnosis').value
        },
        analysis_result: currentResult,
        images: {
            original: originalImageBase64,
            segmented: currentResult.images.segmented
        }
    };

    try {
        exportBtn.disabled = true;
        statusDiv.innerHTML = '<span style="color: blue;">⌛ Đang khởi tạo PDF...</span>';
        
        const response = await fetch(`${API_BASE}/api/export-pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Lỗi từ server khi tạo PDF');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Phieu_Kham_${payload.patient_info.id || 'BN'}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        statusDiv.innerHTML = `<span style="color: green;">✅ Đã xuất file PDF!</span>`;
    } catch (error) {
        console.error('❌ Export error:', error);
        statusDiv.innerHTML = `<span style="color: red;">❌ Lỗi: ${error.message}</span>`;
    } finally {
        exportBtn.disabled = false;
    }
});

function displayColorLegend(colorLegend, angleType) {
    const legendContainer = document.getElementById('legendItems');
    legendContainer.innerHTML = '';
    
    colorLegend.forEach(item => {
        const isHighlight = item.key === 'effusion' || item.key === 'synovium';
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color ${isHighlight ? 'legend-highlight' : ''}" 
                 style="background-color: rgb(${item.color.join(',')})"></div>
            <span>${item.name}</span>
        `;
        legendContainer.appendChild(legendItem);
    });
    
    document.getElementById('colorLegend').style.display = 'block';
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showError(msg) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = msg;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function hideResults() {
    document.getElementById('angleResultCard').style.display = 'none';
    document.getElementById('resultsGrid').style.display = 'none';
    document.getElementById('segmentedImageContainer').style.display = 'none';
    document.getElementById('inflammationCard').style.display = 'none';
    document.getElementById('measurementCard').style.display = 'none';
    document.getElementById('severityCard').style.display = 'none';
    document.getElementById('noResults').style.display = 'block';
}

function showResultPopup(result) {
    const modal = document.getElementById('resultModal');
    
    // 1. Hình ảnh
    if (result.images) {
        document.getElementById('modalImgEnhanced').src = result.images.enhanced || '';
        const segImg = document.getElementById('modalImgSegmented');
        const segBox = document.getElementById('modalImgSegmentedBox');
        
        if (result.images.segmented) {
            segImg.src = result.images.segmented;
            segBox.style.display = 'block';
        } else {
            segBox.style.display = 'none';
        }
    }

    // 2. Chi tiết kết quả
    document.getElementById('modalAngle').textContent = ANGLE_NAMES[result.angle.class] || result.angle.class;
    
    const inflEl = document.getElementById('modalInflammation');
    if (result.inflammation) {
        document.getElementById('modalInflammationRow').style.display = 'flex';
        if (result.inflammation.detected) {
            inflEl.innerHTML = '<span class="badge badge-danger">CÓ VIÊM/THEO DÕI</span>';
        } else {
            inflEl.innerHTML = '<span class="badge badge-success">KHÔNG VIÊM</span>';
        }
    } else {
        document.getElementById('modalInflammationRow').style.display = 'none';
    }

    if (result.measurement) {
        document.getElementById('modalMeasurementRow').style.display = 'flex';
        document.getElementById('modalThickness').textContent = `${result.measurement.thickness_mm} mm`;
    } else {
        document.getElementById('modalMeasurementRow').style.display = 'none';
    }

    if (result.severity) {
        document.getElementById('modalSeverityRow').style.display = 'flex';
        const badge = document.getElementById('modalSeverity');
        badge.textContent = result.severity.severity;
        badge.style.background = result.severity.color;
    } else {
        document.getElementById('modalSeverityRow').style.display = 'none';
    }

    // 3. Chú thích màu sắc (Legend) trong modal
    const legendContainer = document.getElementById('modalLegendContainer');
    if (result.segmentation && result.segmentation.performed && result.segmentation.color_legend) {
        legendContainer.style.display = 'block';
        renderModalLegend(result.segmentation.color_legend);
    } else {
        legendContainer.style.display = 'none';
    }

    // Hiển thị modal
    modal.classList.add('active');
}

function renderModalLegend(colorLegend) {
    const itemsContainer = document.getElementById('modalLegendItems');
    itemsContainer.innerHTML = '';
    
    colorLegend.forEach(item => {
        const isHighlight = item.key === 'effusion' || item.key === 'synovium';
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.innerHTML = `
            <div class="legend-color ${isHighlight ? 'legend-highlight' : ''}" 
                 style="background-color: rgb(${item.color.join(',')})"></div>
            <span>${item.name}</span>
        `;
        itemsContainer.appendChild(legendItem);
    });
}

function closeResultPopup() {
    document.getElementById('resultModal').classList.remove('active');
}

// Event Listeners for Modal
document.getElementById('closeModal').addEventListener('click', closeResultPopup);
document.getElementById('modalViewDetail').addEventListener('click', closeResultPopup);

// Click outside to close
document.getElementById('resultModal').addEventListener('click', (e) => {
    if (e.target.id === 'resultModal') closeResultPopup();
});

// Health check
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const health = await response.json();
        console.log('✅ Server ready:', health);
    } catch (error) {
        showError('Không thể kết nối server. Vui lòng khởi động FastAPI backend.');
    }
});