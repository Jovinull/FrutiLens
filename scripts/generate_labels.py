"""
generate_labels.py
------------------
Converte o dataset Fruits-360 (classificação) em um dataset de detecção YOLO.

ESTRATÉGIA:
    Cada imagem do Fruits-360 possui exatamente UMA fruta sobre fundo branco
    uniforme (R,G,B > 240). Por isso, é possível gerar bounding boxes precisos
    automaticamente via limiarização — sem nenhum trabalho manual.

    O bounding box é obtido encontrando a região mínima que envolve todos os
    pixels que NÃO são brancos (ou seja, que pertencem à fruta).
"""

import os
import shutil
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parâmetros de limiar
# ---------------------------------------------------------------------------
WHITE_THRESHOLD = 240   # Pixels com R,G,B > este valor são considerados fundo
MIN_FRUIT_RATIO = 0.05  # Imagem descartada se a fruta ocupar < 5% da área


def get_bounding_box(img_array: np.ndarray, threshold: int = WHITE_THRESHOLD):
    """
    Retorna o bounding box (x_min, y_min, x_max, y_max) da região não-branca.
    Retorna None se nenhuma região significativa for encontrada.
    """
    # Máscara: True onde o pixel NÃO é branco (pertence à fruta)
    mask = ~np.all(img_array > threshold, axis=2)

    fruit_ratio = mask.sum() / mask.size
    if fruit_ratio < MIN_FRUIT_RATIO:
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    """Converte (x_min, y_min, x_max, y_max) para formato YOLO normalizado."""
    cx = (x_min + x_max) / (2 * img_w)
    cy = (y_min + y_max) / (2 * img_h)
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


def process_split(
    split_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    class_to_idx: dict,
    selected_classes: list | None = None,
    threshold: int = WHITE_THRESHOLD,
):
    """
    Processa um split (Training ou Test) do Fruits-360.

    Args:
        split_dir:       Caminho para Training/ ou Test/
        out_images_dir:  Destino das imagens copiadas
        out_labels_dir:  Destino dos arquivos .txt de label
        class_to_idx:    Mapeamento nome_classe -> índice inteiro
        selected_classes: Lista de classes a incluir (None = todas)
        threshold:       Limiar de branco para segmentação

    Returns:
        (n_ok, n_skip) — imagens processadas e ignoradas
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(split_dir.iterdir())
    n_ok, n_skip = 0, 0

    for class_dir in tqdm(class_dirs, desc=f"  {split_dir.name}", leave=False):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if selected_classes and class_name not in selected_classes:
            continue

        if class_name not in class_to_idx:
            continue

        cls_idx = class_to_idx[class_name]

        for img_path in class_dir.glob("*.jpg"):
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                h, w = arr.shape[:2]

                bbox = get_bounding_box(arr, threshold)
                if bbox is None:
                    n_skip += 1
                    continue

                cx, cy, bw, bh = bbox_to_yolo(*bbox, w, h)

                # Nome único do arquivo de saída
                stem = f"{class_name.replace(' ', '_')}_{img_path.stem}"
                dst_img = out_images_dir / f"{stem}.jpg"
                dst_lbl = out_labels_dir / f"{stem}.txt"

                shutil.copy2(img_path, dst_img)
                dst_lbl.write_text(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                n_ok += 1

            except Exception as exc:
                print(f"  [AVISO] {img_path}: {exc}")
                n_skip += 1

    return n_ok, n_skip


def build_class_map(training_dir: Path, selected_classes: list | None = None) -> dict:
    """Cria o mapeamento nome -> índice ordenado alfabeticamente."""
    classes = sorted(
        d.name for d in training_dir.iterdir() if d.is_dir()
    )
    if selected_classes:
        classes = [c for c in classes if c in selected_classes]
    return {name: idx for idx, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gera labels YOLO automaticamente do Fruits-360"
    )
    parser.add_argument(
        "--fruits360",
        type=str,
        default="dataset/fruits-360_100x100/fruits-360",
        help="Caminho para a raiz do Fruits-360 (contém Training/ e Test/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fruits360_yolo",
        help="Diretório de saída do dataset no formato YOLO",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=(
            "Lista de classes a incluir (padrão: todas). "
            "Exemplo: --classes 'Apple 10' 'Banana 1' 'Orange 1'"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=WHITE_THRESHOLD,
        help=f"Limiar de branco para segmentação (padrão: {WHITE_THRESHOLD})",
    )
    args = parser.parse_args()

    fruits360 = Path(args.fruits360)
    output    = Path(args.output)
    training_dir = fruits360 / "Training"
    test_dir     = fruits360 / "Test"

    if not training_dir.exists():
        raise FileNotFoundError(f"Diretório Training não encontrado: {training_dir}")

    print("=" * 60)
    print("  FrutiLens — Geração automática de labels YOLO")
    print("=" * 60)

    # Mapa de classes
    class_to_idx = build_class_map(training_dir, args.classes)
    n_classes = len(class_to_idx)
    print(f"\n  Classes selecionadas: {n_classes}")

    # Salva classes.txt
    output.mkdir(parents=True, exist_ok=True)
    classes_file = output / "classes.txt"
    idx_to_name  = {v: k for k, v in class_to_idx.items()}
    classes_file.write_text(
        "\n".join(idx_to_name[i] for i in range(n_classes)), encoding="utf-8"
    )
    print(f"  classes.txt salvo em: {classes_file}")

    # Processa Training
    print("\n[1/2] Processando Training...")
    ok_tr, skip_tr = process_split(
        training_dir,
        output / "images" / "train",
        output / "labels" / "train",
        class_to_idx,
        args.classes,
        args.threshold,
    )
    print(f"  OK: {ok_tr:,} | Ignoradas: {skip_tr:,}")

    # Processa Test -> val
    print("\n[2/2] Processando Test...")
    ok_te, skip_te = process_split(
        test_dir,
        output / "images" / "val",
        output / "labels" / "val",
        class_to_idx,
        args.classes,
        args.threshold,
    )
    print(f"  OK: {ok_te:,} | Ignoradas: {skip_te:,}")

    # Gera fruits360.yaml
    yaml_path = output / "fruits360.yaml"
    yaml_content = f"""\
# FrutiLens — Dataset Configuration (gerado automaticamente)
path: {output.resolve().as_posix()}
train: images/train
val:   images/val

nc: {n_classes}
names:
"""
    for i in range(n_classes):
        yaml_content += f"  {i}: '{idx_to_name[i]}'\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n  Dataset YAML salvo em: {yaml_path}")

    print(f"\n{'=' * 60}")
    print(f"  Total processadas: {ok_tr + ok_te:,} imagens")
    print(f"  Total ignoradas  : {skip_tr + skip_te:,} imagens")
    print(f"  Classes          : {n_classes}")
    print("=" * 60)
    print("\nPróximo passo:")
    print(f"  python scripts/train.py --data {yaml_path}\n")


if __name__ == "__main__":
    main()
