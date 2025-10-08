import requests

# Test API đơn giản
def test_api():
    # 1. Test server có hoạt động không
    print("=== Test server ===")
    try:
        response = requests.get("http://localhost:5000/")
        print("✅ Server hoạt động:", response.json())
    except:
        print("❌ Server chưa chạy! Hãy chạy: python simple_api.py")
        return
    
    # 2. Test segment ảnh
    print("\n=== Test segment ảnh ===")
    image_path = "test.png"  # Thay bằng đường dẫn ảnh của bạn
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post("http://localhost:5000/segment", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Segment thành công!")
            print(f"   - Mức độ viêm: {result['inflammation_level']}")
            print(f"   - Tỷ lệ: {result['ratio']}%")
            print(f"   - Kích thước ảnh: {result['image_size']}")
        else:
            print(f"❌ Lỗi: {response.json()}")
            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {image_path}")
        print("   Hãy thay đổi đường dẫn file trong test_simple.py")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_api()