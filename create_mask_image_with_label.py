import os
import json
import base64
from PIL import Image, ImageDraw
import io

def create_class_id_mask(json_dir_path, output_folder, label_mapping):
    """
    JSON 파일에서 클래스 레이블을 정수형 클래스 ID로 매핑한 마스크 이미지를 생성합니다.

    :param json_dir_path: JSON 파일들이 있는 디렉토리 경로
    :param output_folder: 생성된 마스크 이미지를 저장할 폴더 경로
    :param label_mapping: 레이블과 클래스 ID를 매핑한 딕셔너리 (예: {"background": 0, "rice": 1, ...})
    """
    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir_path) if f.endswith('.json')]

    for json_file in json_files:
        json_file_path = os.path.join(json_dir_path, json_file)
        try:
            # JSON 파일 읽기
            with open(json_file_path, "r") as file:
                data = json.load(file)

            # Base64 이미지 데이터 디코딩
            image_data_base64 = data.get("imageData")
            if image_data_base64:
                # 패딩 문제 해결
                padding = len(image_data_base64) % 4
                if padding:
                    image_data_base64 += "=" * (4 - padding)

                image_binary = base64.b64decode(image_data_base64)
                original_image = Image.open(io.BytesIO(image_binary))

                # 흑백 이미지(단일 채널, 클래스 ID 저장용) 생성
                mask_image = Image.new("L", original_image.size, 0)
                draw = ImageDraw.Draw(mask_image)

                for shape in data.get("shapes", []):
                    label = shape["label"]
                    points = shape["points"]

                    # label을 클래스 ID로 매핑
                    class_id = label_mapping.get(label, 0)  # 없는 레이블은 0 (배경)으로 처리

                    # 다각형 그리기
                    polygon = [(x, y) for x, y in points]
                    draw.polygon(polygon, fill=class_id)

                # 저장
                output_path = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}_mask.png")
                mask_image.save(output_path)
                print(f"Class ID mask saved at {output_path}")
            else:
                print(f"No 'imageData' field found in {json_file}.")
        except Exception as e:
            print(f"Failed to process {json_file_path}: {e}")

# 클래스 레이블과 ID 매핑 정의
label_mapping = {
    "background": 0,
    "rice": 1,
    "salad": 2,
    "popcorn chicken": 3,
    "danmuji": 4,
    "donggeurangddaeng": 5
}

# 경로 설정
json_dir_path = "/data/computer_vision_project_seg/dataset/json"  # JSON 파일 경로
output_folder = "computer_vision_project_seg/dataset/class_masked_images"  # 저장 폴더

# 실행
create_class_id_mask(json_dir_path, output_folder, label_mapping)
