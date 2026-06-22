#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4개의 이미지를 2x2 레이아웃으로 합치는 스크립트

사용 예시:
python combine_2x2_graphs.py \
  --img1 bi_lstm.png \
  --img2 lstm.png \
  --img3 tcn.png \
  --img4 transformer.png \
  --output combined_2x2.png

옵션:
- 모든 이미지를 동일 크기로 맞춘 뒤 2x2로 배치
- 여백(margin)과 내부 간격(gap) 설정 가능
- 필요 시 외곽 테두리 추가 가능
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


def combine_images_2x2(
    img1_path: Path,
    img2_path: Path,
    img3_path: Path,
    img4_path: Path,
    output_path: Path,
    margin: int = 20,
    gap: int = 20,
    add_border: bool = False,
) -> Path:
    """
    이미지를 아래 순서로 2x2 배치한다.
    [img1, img2]
    [img3, img4]
    """
    # 첫 번째 이미지를 기준으로 크기 통일
    base_img = Image.open(img1_path).convert("RGB")
    target_w, target_h = base_img.size

    img1 = load_and_resize(img1_path, (target_w, target_h), add_border=add_border)
    img2 = load_and_resize(img2_path, (target_w, target_h), add_border=add_border)
    img3 = load_and_resize(img3_path, (target_w, target_h), add_border=add_border)
    img4 = load_and_resize(img4_path, (target_w, target_h), add_border=add_border)

    # border가 추가되었으면 실제 크기 재계산
    tile_w, tile_h = img1.size

    canvas_w = margin * 2 + tile_w * 2 + gap
    canvas_h = margin * 2 + tile_h * 2 + gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Paste positions
    positions = [
        (margin, margin),                                  # top-left
        (margin + tile_w + gap, margin),                   # top-right
        (margin, margin + tile_h + gap),                   # bottom-left
        (margin + tile_w + gap, margin + tile_h + gap),    # bottom-right
    ]

    for img, pos in zip([img1, img2, img3, img4], positions):
        canvas.paste(img, pos)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Combine 4 graph images into one 2x2 image.")
    parser.add_argument("--img1", required=True, help="Top-left image path")
    parser.add_argument("--img2", required=True, help="Top-right image path")
    parser.add_argument("--img3", required=True, help="Bottom-left image path")
    parser.add_argument("--img4", required=True, help="Bottom-right image path")
    parser.add_argument("--output", default="combined_2x2.png", help="Output image path")
    parser.add_argument("--margin", type=int, default=20, help="Outer margin size")
    parser.add_argument("--gap", type=int, default=20, help="Gap between images")
    parser.add_argument("--add-border", action="store_true", help="Add black border to each image")
    args = parser.parse_args()

    output = combine_images_2x2(
        img1_path=Path(args.img1),
        img2_path=Path(args.img2),
        img3_path=Path(args.img3),
        img4_path=Path(args.img4),
        output_path=Path(args.output),
        margin=args.margin,
        gap=args.gap,
        add_border=args.add_border,
    )

    print(f"Saved combined image to: {output}")


if __name__ == "__main__":
    main()
