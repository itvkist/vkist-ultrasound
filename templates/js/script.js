const API_BASE = 'http://localhost:8000';
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

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
        document.getElementById('originalImage').src = e.target.result;
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
        console.log('✅ Response:', result);
        
        if (result.success) displayResults(result);
        else showError('Phân tích thất bại');

    } catch (error) {
        console.error('❌ Error:', error);
        showError(`Lỗi: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    document.getElementById('noResults').style.display = 'none';
    
    document.getElementById('angleResultCard').style.display = 'block';
    document.getElementById('angleValue').textContent = ANGLE_NAMES[result.angle.class] || result.angle.class;
    document.getElementById('angleConfidenceText').textContent = `Độ tin cậy: ${result.angle.confidence}%`;

    if (result.inflammation) {
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
        document.getElementById('segmentedImageContainer').style.display = 'block';
        document.getElementById('segmentedImage').src = API_BASE + result.images.segmented;
        
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
        document.getElementById('measurementCard').style.display = 'block';
        
        document.getElementById('thicknessMm').textContent = `${result.measurement.thickness_mm} mm`;
        document.getElementById('thicknessPx').textContent = `${result.measurement.thickness_px}`;
        document.getElementById('measurementLocationX').textContent = result.measurement.location_x;
        document.getElementById('measurementRangeY').textContent = 
            `${result.measurement.y_start} - ${result.measurement.y_end}`;
        
        console.log('📏 Measurement displayed:', result.measurement);
    }

    if (result.severity) {
        document.getElementById('severityCard').style.display = 'block';
        const badge = document.getElementById('severityBadge');
        badge.textContent = result.severity.severity;
        badge.style.background = result.severity.color;
        document.getElementById('severityDescription').textContent = result.severity.description;
        document.getElementById('effusionStat').textContent = 
            `${result.severity.effusion.ratio}% (${result.severity.effusion.thickness}px)`;
        document.getElementById('synoviumStat').textContent = `${result.severity.synovium.ratio}%`;
    }
}

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