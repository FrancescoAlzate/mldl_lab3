"""
Script per preparare il dataset TinyImageNet-200.
Organizza le immagini di validazione nelle cartelle appropriate.
"""
import os
import shutil
import argparse
from pathlib import Path


def prepare_tiny_imagenet_val(data_root: str) -> None:
    """
    Prepara il dataset TinyImageNet organizzando le immagini di validazione.
    
    Args:
        data_root: Path alla root del dataset (es. 'tiny-imagenet/tiny-imagenet-200')
    """
    val_dir = os.path.join(data_root, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    
    if not os.path.exists(val_annotations_file):
        raise FileNotFoundError(f"File {val_annotations_file} non trovato. Assicurati che il dataset sia stato scaricato.")
    
    if not os.path.exists(val_images_dir):
        print(f"Cartella {val_images_dir} non trovata. Il dataset potrebbe essere già stato preparato.")
        return
    
    print(f"Preparazione dataset in {data_root}...")
    
    # Leggi le annotazioni e organizza le immagini
    with open(val_annotations_file, 'r') as f:
        for line in f:
            fn, cls, *_ = line.strip().split('\t')
            target_dir = os.path.join(val_dir, cls)
            os.makedirs(target_dir, exist_ok=True)
            
            source_path = os.path.join(val_images_dir, fn)
            target_path = os.path.join(target_dir, fn)
            
            if os.path.exists(source_path):
                shutil.copyfile(source_path, target_path)
    
    # Rimuovi la cartella images se esiste
    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)
        print(f"Rimossa cartella {val_images_dir}")
    
    print("Dataset preparato con successo!")


def main():
    parser = argparse.ArgumentParser(description="Prepara il dataset TinyImageNet-200")
    parser.add_argument(
        "--data-root",
        type=str,
        default="tiny-imagenet/tiny-imagenet-200",
        help="Path alla root del dataset TinyImageNet-200"
    )
    args = parser.parse_args()
    
    prepare_tiny_imagenet_val(args.data_root)


if __name__ == "__main__":
    main()

