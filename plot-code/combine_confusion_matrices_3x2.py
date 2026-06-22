#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6개의 이미지를 3x2 레이아웃으로 합치는 스크립트

배치 순서:
[ img1 | img2 ]
[ img3 | img4 ]
[ img5 | img6 ]

사용 예시:
python combine_confusion_matrices_3x2.py \
  --img1 random_forest.png \
  --img2 bi_lstm.png \
  --img3 lstm.png \
  --img4 tcn.png \
  --img5 transformer.png \
  --img6 xgboost.png \
  --output combined_confusion_matrices_3x2.png \
  --dpi 300

기능:
- 6개 이미지를 동일 크기로 맞춘 뒤 3x2로 배치
- 외곽 여백(margin), 이미지 간 간격(gap) 설정 가능
- 선택적으로 각 이미지에 검은 테두리 추가 가능
- 저장 시 DPI 지정 가능
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps


def load_and_resize(image_path: Path, target_size: tuple[int, int], add_border: bool = False) -> Image.Image:
    """이미지를 불러와 target_size로 맞춘다."""
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)

    if add_border:
        img = ImageOps.expand(img, border=2, fill="black")

    return img


def combine_images_3x2(
    image_paths: list[Path],
    output_path: Path,
    margin: int = 20,
    gap: int = 20,
    add_border: bool = False,
    dpi: int = 300,
) -> Path:
    """
    6개의 이미지를 아래 순서로 3x2 배치한다.
    [img1, img2]
    [img3, img4]
    [img5, img6]
    """
    if len(image_paths) != 6:
        raise ValueError("image_paths에는 정확히 6개의 이미지 경로가 필요합니다.")

    for p in image_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {p}")

    # 첫 번째 이미지를 기준으로 크기 통일
    base_img = Image.open(image_paths[0]).convert("RGB")
    target_w, target_h = base_img.size

    images = [
        load_and_resize(Path(p), (target_w, target_h), add_border=add_border)
        for p in image_paths
    ]

    tile_w, tile_h = images[0].size

    canvas_w = margin * 2 + tile_w * 2 + gap
    canvas_h = margin * 2 + tile_h * 3 + gap * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    positions = [
        (margin, margin),                                      # row 1 col 1
        (margin + tile_w + gap, margin),                       # row 1 col 2
        (margin, margin + tile_h + gap),                       # row 2 col 1
        (margin + tile_w + gap, margin + tile_h + gap),        # row 2 col 2
        (margin, margin + (tile_h + gap) * 2),                 # row 3 col 1
        (margin + tile_w + gap, margin + (tile_h + gap) * 2),  # row 3 col 2
    ]

    for img, pos in zip(images, positions):
        canvas.paste(img, pos)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # dpi=300으로 저장
    canvas.save(output_path, dpi=(dpi, dpi))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Combine 6 images into one 3x2 image.")
    parser.add_argument("--img1", required=True, help="Row1-Col1 image path")
    parser.add_argument("--img2", required=True, help="Row1-Col2 image path")
    parser.add_argument("--img3", required=True, help="Row2-Col1 image path")
    parser.add_argument("--img4", required=True, help="Row2-Col2 image path")
    parser.add_argument("--img5", required=True, help="Row3-Col1 image path")
    parser.add_argument("--img6", required=True, help="Row3-Col2 image path")
    parser.add_argument("--output", default="combined_confusion_matrices_3x2.png", help="Output image path")
    parser.add_argument("--margin", type=int, default=20, help="Outer margin size")
    parser.add_argument("--gap", type=int, default=20, help="Gap between images")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--add-border", action="store_true", help="Add black border to each image")
    args = parser.parse_args()

    image_paths = [
        Path(args.img1),
        Path(args.img2),
        Path(args.img3),
        Path(args.img4),
        Path(args.img5),
        Path(args.img6),
    ]

    output = combine_images_3x2(
        image_paths=image_paths,
        output_path=Path(args.output),
        margin=args.margin,
        gap=args.gap,
        add_border=args.add_border,
        dpi=args.dpi,
    )

    print(f"Saved combined image to: {output}")


if __name__ == "__main__":
    main()
