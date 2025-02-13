import os
import json
import base64
from PIL import Image, ImageDraw
import io

def create_colored_mask_with_labels(json_dir_path, output_folder):
    """
    JSON 파일에서 동일한 객체(label)를 동일한 색상으로 표시한 마스크 이미지를 생성합니다.

    :param json_dir_path: JSON 파일들이 있는 디렉토리 경로
    :param output_folder: 생성된 마스크 이미지를 저장할 폴더 경로
    """
    os.makedirs(output_folder, exist_ok=True)

    # 색상 팔레트 생성
    colors = [
        (255, 0, 0, 255),    # 빨강
        (0, 255, 0, 255),    # 초록
        (0, 0, 255, 255),    # 파랑
        (255, 255, 0, 255),  # 노랑
        (128, 0, 128, 255),  # 보라
        (255, 0, 255, 255),  # 자홍
    ]

    # Label 별 색상 매핑
    label_color_map = {}

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

                # 투명한 배경 위에 객체 색상 표시
                mask_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask_image)

                for shape in data.get("shapes", []):
                    label = shape["label"]
                    points = shape["points"]

                    # 동일한 label에 대해 동일한 색상을 유지
                    if label not in label_color_map:
                        # 새로운 label에 색상 할당
                        color = colors[len(label_color_map) % len(colors)]
                        label_color_map[label] = color
                    else:
                        color = label_color_map[label]

                    # 다각형 그리기
                    polygon = [(x, y) for x, y in points]
                    draw.polygon(polygon, fill=color)

                # 저장
                output_path = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}_mask.png")
                mask_image.save(output_path)
                print(f"Mask image saved at {output_path}")
            else:
                print(f"No 'imageData' field found in {json_file}.")
        except Exception as e:
            print(f"Failed to process {json_file_path}: {e}")

# 경로 설정
json_dir_path = "C:/Users/jhpark/Desktop/computer_vision/segmentation_dataset/labels"  # JSON 파일 경로
output_folder = "C:/Users/jhpark/Desktop/computer_vision/segmentation_dataset/masked_images"  # 저장 폴더

# 실행
create_colored_mask_with_labels(json_dir_path, output_folder)
