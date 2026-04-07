"""
train.py
--------
Treina YOLOv8 no dataset Fruits-360 convertido para detecção.

Uso básico:
    python scripts/train.py

Uso avançado:
    python scripts/train.py --data data/fruits360_yolo/fruits360.yaml \\
                            --model yolov8s.pt \\
                            --epochs 100 \\
                            --imgsz 320 \\
                            --batch 32 \\
                            --name frutilens_v1
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Treinamento YOLOv8 — FrutiLens")

    parser.add_argument(
        "--data",
        type=str,
        default="data/fruits360_yolo/fruits360.yaml",
        help="Caminho para o arquivo .yaml do dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help=(
            "Modelo base para fine-tuning. Opções:\n"
            "  yolov8n.pt  — Nano   (~3.2M params) — mais rápido, menor precisão\n"
            "  yolov8s.pt  — Small  (~11M params)  — bom equilíbrio\n"
            "  yolov8m.pt  — Medium (~26M params)  — alta precisão\n"
            "  yolov8l.pt  — Large  (~44M params)  — máxima precisão, mais lento"
        ),
    )
    parser.add_argument("--epochs",  type=int, default=50,  help="Número de épocas")
    parser.add_argument("--imgsz",   type=int, default=320, help="Tamanho da imagem (pixels)")
    parser.add_argument("--batch",   type=int, default=32,  help="Batch size (-1 = automático)")
    parser.add_argument("--workers", type=int, default=4,   help="DataLoader workers")
    parser.add_argument("--device",  type=str, default="",  help="Dispositivo: '' (auto), 'cpu', '0', '0,1'")
    parser.add_argument("--name",    type=str, default="frutilens", help="Nome do experimento")
    parser.add_argument("--patience",type=int, default=20, help="Early stopping: épocas sem melhora")
    parser.add_argument("--resume",  action="store_true",   help="Retomar treinamento anterior")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n[ERRO] Dataset não encontrado: {data_path}")
        print("Execute primeiro:")
        print("  python scripts/generate_labels.py")
        return

    print("=" * 60)
    print("  FrutiLens — Treinamento YOLOv8")
    print("=" * 60)
    print(f"  Modelo   : {args.model}")
    print(f"  Dataset  : {args.data}")
    print(f"  Épocas   : {args.epochs}")
    print(f"  Imagem   : {args.imgsz}x{args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device or 'automático'}")
    print("=" * 60 + "\n")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device else None,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # Augmentações recomendadas para ambiente industrial
        hsv_h=0.015,    # variação de matiz
        hsv_s=0.7,      # variação de saturação
        hsv_v=0.4,      # variação de brilho
        flipud=0.0,      # sem flip vertical (esteira = orientação definida)
        fliplr=0.5,     # flip horizontal (frutas podem vir em ambos lados)
        mosaic=1.0,     # mosaic augmentation (muito útil para múltiplas frutas)
        mixup=0.1,
        copy_paste=0.1,
        # Otimizador
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
    )

    print("\n" + "=" * 60)
    print("  Treinamento concluído!")
    best_weights = Path("runs/detect") / args.name / "weights/best.pt"
    print(f"  Melhores pesos: {best_weights}")
    print("\nPróximo passo:")
    print(f"  python src/detect.py --weights {best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
