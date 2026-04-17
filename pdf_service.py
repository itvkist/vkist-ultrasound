from fpdf import FPDF
import io
import base64
import os
from datetime import datetime
from PIL import Image

class MedicalReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.main_font = 'helvetica' # Default fallback
        self.set_margins(10, 10, 10)

    def header(self):
        # Logo support
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'logo.png')
        show_logo = False
        
        if os.path.exists(logo_path):
            logo_stream = get_clean_image_stream(logo_path)
            if logo_stream:
                try:
                    self.image(logo_stream, 10, 8, 25, type='PNG')
                    self.set_x(40)
                    show_logo = True
                except:
                    pass
        
        # Header with Unicode support
        try:
            self.set_font(self.main_font, 'B', 16)
            title = 'TRUNG TÂM CHẨN ĐOÁN HÌNH ẢNH VKIST'
            if show_logo:
                self.cell(0, 10, title, 0, 1, 'L')
                self.set_x(40)
                self.set_font(self.main_font, '', 10)
                self.cell(0, 5, 'Địa chỉ: Khu Công nghệ cao Hòa Lạc, Thạch Thất, Hà Nội', 0, 1, 'L')
            else:
                self.cell(0, 10, title, 0, 1, 'C')
                self.set_font(self.main_font, '', 10)
                self.cell(0, 10, 'Địa chỉ: Khu Công nghệ cao Hòa Lạc, Thạch Thất, Hà Nội', 0, 1, 'C')
            self.ln(10)
        except Exception:
            pass

    def footer(self):
        self.set_y(-15)
        try:
            self.set_font(self.main_font, 'I', 8)
            self.cell(0, 10, f'Trang {self.page_no()}/{{nb}}', 0, 0, 'C')
        except Exception:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def get_clean_image_stream(image_source):
    """
    Mở ảnh (file path hoặc base64), loại bỏ interlacing và trả về BytesIO stream.
    Giúp tránh lỗi 'Interlacing not supported' trong FPDF2.
    """
    if not image_source:
        return None
        
    try:
        if isinstance(image_source, str) and image_source.startswith('data:image'):
            # Xử lý base64
            header, content = image_source.split(',') if ',' in image_source else (None, image_source)
            img_data = base64.b64decode(content)
            img_io = io.BytesIO(img_data)
        elif isinstance(image_source, str) and os.path.exists(image_source):
            # Xử lý file path
            img_io = image_source
        else:
            return None

        with Image.open(img_io) as img:
            # Chuyển đổi sang RGB nếu cần và lưu lại không có interlacing
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=False, interlaced=False)
            output.seek(0)
            return output
    except Exception as e:
        print(f"⚠️ Lỗi xử lý ảnh: {e}")
        return None

def generate_medical_report(patient_info, analysis_result, images_base64):
    pdf = MedicalReportPDF()
    
    # Sử dụng đường dẫn tuyệt đối để ổn định
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(base_dir, 'assets', 'fonts')
    
    # 1. Đăng ký Font nội bộ (Arial đã sao chép)
    try:
        pdf.add_font('arial_local', '', os.path.join(font_dir, 'arial.ttf'))
        pdf.add_font('arial_local', 'B', os.path.join(font_dir, 'arialbd.ttf'))
        pdf.add_font('arial_local', 'I', os.path.join(font_dir, 'ariali.ttf'))
        pdf.main_font = 'arial_local'
    except Exception as e:
        print(f"Warning: Could not load local Arial fonts: {e}")
        # Dự phòng cuối cùng
        pdf.main_font = 'helvetica'
    
    font_main = pdf.main_font
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.alias_nb_pages()

    # Tiêu đề
    pdf.set_font(font_main, 'B', 14)
    pdf.cell(0, 10, 'PHIẾU KẾT QUẢ SIÊU ÂM KHỚP GỐI', 0, 1, 'C')
    pdf.ln(5)
    
    # Thông tin bệnh nhân
    pdf.set_x(10)
    pdf.set_font(font_main, 'B', 11)
    pdf.cell(0, 8, 'I. THÔNG TIN BỆNH NHÂN', 0, 1, 'L')
    pdf.set_font(font_main, '', 11)
    
    # Độ rộng cột an toàn (Tổng 180mm < 190mm khả dụng cho A4)
    col1 = 95
    col2 = 85
    
    pdf.cell(col1, 8, f"Họ tên: {patient_info.get('name', 'N/A')}", 0, 0)
    pdf.cell(col2, 8, f"Mã BN: {patient_info.get('id', 'N/A')}", 0, 1)
    
    pdf.cell(col1, 8, f"Giới tính: {patient_info.get('gender', 'N/A')}", 0, 0)
    pdf.cell(col2, 8, f"Tuổi: {patient_info.get('age', 'N/A')}", 0, 1)
    
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Images Section
    pdf.set_x(10)
    pdf.set_font(font_main, 'B', 11)
    pdf.cell(0, 10, 'II. HÌNH ẢNH SIÊU ÂM', 0, 1, 'L')
    
    y_before_images = pdf.get_y()
    img_w = 90
    margin = 5
    
    # Process Images
    has_orig = images_base64.get('original')
    has_seg = images_base64.get('segmented')
    
    max_img_h = 0
    
    if has_orig:
        try:
            orig_stream = get_clean_image_stream(has_orig)
            if orig_stream:
                with Image.open(orig_stream) as img:
                    w, h = img.size
                    img_h = (img_w * h) / w
                    max_img_h = max(max_img_h, img_h)
                
                orig_stream.seek(0)
                pdf.image(orig_stream, x=10, y=y_before_images, w=img_w, type='PNG')
                
                # Label
                pdf.set_xy(10, y_before_images + img_h + 2)
                pdf.set_font(font_main, 'I', 9)
                pdf.cell(img_w, 5, 'Hình 1: Ảnh gốc / Tăng cường', 0, 0, 'C')
        except Exception as e:
            print(f"Error processing original image: {e}")

    if has_seg:
        try:
            seg_stream = get_clean_image_stream(has_seg)
            if seg_stream:
                with Image.open(seg_stream) as img:
                    w, h = img.size
                    img_h = (img_w * h) / w
                    max_img_h = max(max_img_h, img_h)
                
                seg_stream.seek(0)
                pdf.image(seg_stream, x=110, y=y_before_images, w=img_w, type='PNG')
                
                # Label
                pdf.set_xy(110, y_before_images + img_h + 2)
                pdf.set_font(font_main, 'I', 9)
                pdf.cell(img_w, 5, 'Hình 2: Ảnh phân đoạn AI', 0, 1, 'C')
        except Exception as e:
            print(f"Error processing segmented image: {e}")

    # Reset Y to after images
    pdf.set_y(y_before_images + max_img_h + 10)
    pdf.set_x(10)
    
    # AI Results
    pdf.set_font(font_main, 'B', 11)
    pdf.cell(0, 10, 'III. KẾT QUẢ PHÂN TÍCH TỰ ĐỘNG (AI)', 0, 1, 'L')
    pdf.set_font(font_main, '', 10)
    
    angle = analysis_result.get('angle', {})
    pdf.multi_cell(185, 6, f"• Góc chụp dự đoán: {angle.get('class', 'N/A')} (Độ tin cậy: {angle.get('confidence', 'N/A')}%)")
    
    if 'inflammation' in analysis_result:
        infl = analysis_result['inflammation']
        status = "Có khả năng viêm / Theo dõi viêm" if infl.get('detected') else "Không thấy dấu hiệu viêm rõ rệt"
        pdf.set_x(10)
        pdf.multi_cell(185, 6, f"• Tình trạng viêm: {status} (Độ tin cậy: {infl.get('confidence', 'N/A')}%)")
        
    if 'measurement' in analysis_result:
        m = analysis_result['measurement']
        pdf.set_x(10)
        pdf.multi_cell(185, 6, f"• Đo đạc: Độ dày dịch & màng hoạt dịch đạt mức {m.get('thickness_mm', 'N/A')} mm")
        
    if 'severity' in analysis_result:
        s = analysis_result['severity']
        pdf.set_x(10)
        pdf.multi_cell(185, 6, f"• Mức độ viêm: {s.get('severity', 'N/A')}")
        pdf.set_x(10)
        pdf.set_font(font_main, 'I', 10)
        pdf.multi_cell(185, 6, f"  Chi tiết: {s.get('description', 'N/A')}")
        pdf.set_font(font_main, '', 10)

    pdf.ln(5)
    pdf.set_x(10)
    
    # Doctor Diagnosis
    pdf.set_font(font_main, 'B', 11)
    pdf.cell(0, 10, 'IV. CHẨN ĐOÁN VÀ KẾT LUẬN CỦA BÁC SĨ', 0, 1, 'L')
    pdf.set_font(font_main, '', 11)
    
    diagnosis = patient_info.get('diagnosis', 'Ghi chú chẩn đoán trống.')
    pdf.set_x(10)
    pdf.multi_cell(185, 7, diagnosis)
    
    pdf.ln(15)
    
    # Signature
    current_date = datetime.now()
    date_str = f"Ngày {current_date.day} tháng {current_date.month} năm {current_date.year}"
    
    pdf.set_font(font_main, 'I', 11)
    pdf.cell(0, 8, date_str, 0, 1, 'R')
    pdf.set_font(font_main, 'B', 11)
    pdf.cell(0, 8, 'BÁC SĨ CHẨN ĐOÁN', 0, 1, 'R')
    pdf.ln(15)
    pdf.set_font(font_main, '', 10)
    pdf.cell(0, 8, '(Ký và ghi rõ họ tên)', 0, 1, 'R')

    return pdf.output()
