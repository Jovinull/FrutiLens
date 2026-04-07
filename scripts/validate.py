"""
validate.py
-----------
Avalia o modelo treinado no conjunto de validação e exibe métricas detalhadas.

Uso:
    python scripts/validate.py --weights runs/detect/frutilens/weights/best.pt \\
                               --data data/fruits360_yolo/fruits360.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validação YOLOv8 — FrutiLens")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para o arquivo .pt dos pesos treinados",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/fruits360_yolo/fruits360.yaml",
        help="Caminho para o .yaml do dataset",
    )
    parser.add_argument("--imgsz",  type=int,   default=320,  help="Tamanho da imagem")
    parser.add_argument("--batch",  type=int,   default=32,   help="Batch size")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confiança mínima")
    parser.add_argument("--iou",    type=float, default=0.6,  help="IoU para NMS")
    parser.add_argument("--device", type=str,   default="",   help="Dispositivo")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERRO] Pesos não encontrados: {weights}")
        return

    print("=" * 60)
    print("  FrutiLens — Validação do Modelo")
    print("=" * 60)

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
    )

    print("\n" + "=" * 60)
    print("  Resultados:")
    print(f"  mAP@0.50      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95 : {metrics.box.map:.4f}")
    print(f"  Precisão      : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
