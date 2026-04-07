"""
detect.py
---------
Detecção de frutas em tempo real usando YOLOv8.
Suporta webcam, arquivo de vídeo e imagem estática.

Uso:
    # Webcam (padrão)
    python src/detect.py --weights runs/detect/frutilens/weights/best.pt

    # Arquivo de vídeo
    python src/detect.py --weights best.pt --source video.mp4

    # Imagem
    python src/detect.py --weights best.pt --source imagem.jpg

    # Salvar resultado
    python src/detect.py --weights best.pt --source video.mp4 --save
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


# ---------------------------------------------------------------------------
# Utilitários de visualização
# ---------------------------------------------------------------------------

def draw_fps(frame, fps: float):
    """Exibe o FPS no canto superior esquerdo."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_counter(frame, counts: dict, frame_w: int):
    """Exibe o contador de frutas detectadas no canto superior direito."""
    y = 30
    for name, count in sorted(counts.items()):
        text = f"{name}: {count}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            frame,
            text,
            (frame_w - tw - 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25


def draw_detections(frame, boxes, clss, confs, names):
    """
    Desenha bounding boxes, rótulos e confiança para cada detecção.

    Returns:
        dict: contagem de cada classe detectada no frame
    """
    counts = {}

    for box, cls, conf in zip(boxes, clss, confs):
        cls_int = int(cls)
        color   = colors(cls_int, True)

        # Bounding box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Rótulo com fundo colorido para melhor legibilidade
        label   = f"{names[cls_int]}  {conf:.0%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Contagem
        name = names[cls_int]
        counts[name] = counts.get(name, 0) + 1

    return counts


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(
    weights: str,
    source,
    imgsz: int = 640,
    conf: float = 0.5,
    iou: float = 0.45,
    device: str = "",
    save: bool = False,
    save_path: str = "output.mp4",
    show: bool = True,
):
    """
    Executa a detecção.

    Args:
        weights:    Caminho para o arquivo .pt
        source:     0 (webcam), caminho de vídeo ou imagem
        imgsz:      Tamanho de entrada da rede
        conf:       Limiar de confiança
        iou:        Limiar de IoU para NMS
        device:     '' (auto), 'cpu', '0'
        save:       Salvar vídeo de saída
        save_path:  Caminho do arquivo de saída
        show:       Exibir janela
    """
    # Carrega modelo
    print(f"[INFO] Carregando modelo: {weights}")
    model = YOLO(weights)
    if device:
        model.to(device)

    # Abre fonte de vídeo/imagem
    is_image = isinstance(source, str) and Path(source).suffix.lower() in (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    )

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Imagem não encontrada: {source}")

        results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss  = results[0].boxes.cls.cpu().tolist()
        confs_list = results[0].boxes.conf.cpu().tolist()

        counts = draw_detections(frame, boxes, clss, confs_list, model.names)
        draw_counter(frame, counts, frame.shape[1])

        out_path = save_path if save_path.endswith((".jpg", ".png")) else "output.jpg"
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Resultado salvo em: {out_path}")

        if show:
            cv2.imshow("FrutiLens", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Vídeo / webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configura resolução para webcam
    if source == 0 or source == "0":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        frame_w, frame_h = 1280, 720

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer  = cv2.VideoWriter(save_path, fourcc, fps_out, (frame_w, frame_h))
        print(f"[INFO] Gravando saída em: {save_path}")

    print("[INFO] Iniciando detecção. Pressione 'q' para sair.")

    # Métricas de FPS (média móvel)
    fps_history: list[float] = []
    t_prev = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Inferência
        results = model.predict(
            source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False
        )

        boxes      = results[0].boxes.xyxy.cpu().tolist()
        clss       = results[0].boxes.cls.cpu().tolist()
        confs_list = results[0].boxes.conf.cpu().tolist()

        # Visualização
        counts = draw_detections(frame, boxes, clss, confs_list, model.names)

        t_now = time.perf_counter()
        fps_history.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        draw_fps(frame, avg_fps)
        draw_counter(frame, counts, frame_w)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("FrutiLens — Detecção em Tempo Real", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Detecção encerrada.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FrutiLens — Detecção de frutas em tempo real (YOLOv8)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para o arquivo de pesos .pt",
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Fonte: 0 (webcam), caminho de vídeo ou imagem (padrão: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho de entrada do modelo (padrão: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confiança mínima para detecção (padrão: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Limiar de IoU para Non-Maximum Suppression (padrão: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo: '' (automático), 'cpu', '0' (GPU) (padrão: automático)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar vídeo/imagem com as detecções",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Caminho do arquivo de saída quando --save é usado (padrão: output.mp4)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não exibir janela (útil para execução headless)",
    )

    args = parser.parse_args()

    # Converte source para int se for webcam
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    run(
        weights=args.weights,
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
