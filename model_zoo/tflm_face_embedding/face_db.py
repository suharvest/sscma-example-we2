#!/usr/bin/env python3
"""
Face database management — register and recognize faces using the Himax-equivalent
embedding pipeline.

Usage:
  # Register a face
  python face_db.py register --name "John" photo.jpg
  python face_db.py register-device-json --name "John" /tmp/himax_facedbg.json
  python face_db.py register --name "Jane" --db my_faces.json photo.jpg

  # Recognize a face
  python face_db.py recognize photo.jpg
  python face_db.py recognize --db my_faces.json --threshold 0.5 photo.jpg

  # List registered faces
  python face_db.py list

  # Delete a face
  python face_db.py delete --name "John"

  # Export database for device (embeddings as binary)
  python face_db.py export --output faces.bin
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Import the pipeline from compute_embedding
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compute_embedding import (
    FaceEmbeddingPipeline,
    EMB_OUTPUT_DIM,
    cosine_similarity,
    SCRFD_TFLITE,
    MOBILEFACENET_FLOAT32_TFLITE,
    MOBILEFACENET_INT8_TFLITE,
    GHOSTFACENET_INT8_TFLITE,
    SCRFD_ONNX,
    MOBILEFACENET_ONNX,
    l2_normalize,
)

DEFAULT_DB = SCRIPT_DIR / "face_database.json"
DEFAULT_THRESHOLD = 0.45


class FaceDatabase:
    """Simple JSON-based face embedding database."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.faces: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                self.faces = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.faces, f, indent=2)

    def add(self, name: str, embedding: list, metadata: dict):
        """Register a face. Overwrites if name exists."""
        self.faces[name] = {
            "embedding": embedding,
            "metadata": metadata,
        }
        self._save()
        print(f"Registered: {name}")

    def delete(self, name: str) -> bool:
        if name in self.faces:
            del self.faces[name]
            self._save()
            print(f"Deleted: {name}")
            return True
        print(f"Not found: {name}")
        return False

    def list(self) -> list[str]:
        return sorted(self.faces.keys())

    def recognize(self, embedding: np.ndarray, threshold: float) -> Optional[dict]:
        """Find the closest registered face above threshold."""
        best_name = None
        best_sim = -1.0
        for name, entry in self.faces.items():
            stored = np.array(entry["embedding"], dtype=np.float32)
            sim = cosine_similarity(embedding, stored)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return {"name": best_name, "similarity": float(best_sim)}
        return None

    def to_binary(self) -> bytes:
        """Export all embeddings as packed binary (for device upload).

        Format: [num_faces: u32][dim: u32] + for each: [name_len: u16][name: utf8][embedding: f32*128]
        """
        import struct

        entries = list(self.faces.items())
        buf = bytearray()
        buf.extend(struct.pack("<II", len(entries), EMB_OUTPUT_DIM))
        for name, entry in entries:
            name_bytes = name.encode("utf-8")
            buf.extend(struct.pack("<H", len(name_bytes)))
            buf.extend(name_bytes)
            emb = np.array(entry["embedding"], dtype=np.float32)
            buf.extend(emb.tobytes())
        return bytes(buf)


def create_pipeline(backend: str, scrfd_model: str, emb_model: str) -> FaceEmbeddingPipeline:
    """Create pipeline with optional model overrides."""
    return FaceEmbeddingPipeline(scrfd_model, emb_model, backend)


def main():
    parser = argparse.ArgumentParser(
        description="Face database — register and recognize faces"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # register
    reg = sub.add_parser("register", help="Register a face from an image")
    reg.add_argument("image", help="Path to face image")
    reg.add_argument("--name", "-n", required=True, help="Person name")
    reg.add_argument("--db", default=str(DEFAULT_DB), help="Database path")
    reg.add_argument("--backend", choices=["tflite", "onnx"], default="tflite")
    reg.add_argument("--scrfd-model", help="Override SCRFD model")
    reg.add_argument("--embedding-model", help="Override embedding model")
    reg.add_argument("--float32-embedding", action="store_true",
                     help="Use float32 embedding model for research/offline comparison")
    reg.add_argument("--debug", "-d", action="store_true")

    # register-device-json
    reg_dev = sub.add_parser(
        "register-device-json",
        help="Register a face from a device INVOKE/FACEDBG capture JSON",
    )
    reg_dev.add_argument("json", help="Path produced by tools/capture_facedbg.py")
    reg_dev.add_argument("--name", "-n", required=True, help="Person name")
    reg_dev.add_argument("--db", default=str(DEFAULT_DB), help="Database path")

    # recognize
    rec = sub.add_parser("recognize", help="Recognize a face from an image")
    rec.add_argument("image", help="Path to face image")
    rec.add_argument("--db", default=str(DEFAULT_DB), help="Database path")
    rec.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                     help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    rec.add_argument("--backend", choices=["tflite", "onnx"], default="tflite")
    rec.add_argument("--scrfd-model", help="Override SCRFD model")
    rec.add_argument("--embedding-model", help="Override embedding model")
    rec.add_argument("--float32-embedding", action="store_true",
                     help="Use float32 embedding model for research/offline comparison")
    rec.add_argument("--debug", "-d", action="store_true")

    # list
    lst = sub.add_parser("list", help="List registered faces")
    lst.add_argument("--db", default=str(DEFAULT_DB), help="Database path")

    # delete
    dlt = sub.add_parser("delete", help="Delete a registered face")
    dlt.add_argument("--name", "-n", required=True, help="Person name")
    dlt.add_argument("--db", default=str(DEFAULT_DB), help="Database path")

    # export
    exp = sub.add_parser("export", help="Export database as binary")
    exp.add_argument("--db", default=str(DEFAULT_DB), help="Database path")
    exp.add_argument("--output", "-o", required=True, help="Output binary file")

    args = parser.parse_args()

    # Commands that don't need the pipeline
    if args.command == "list":
        db = FaceDatabase(args.db)
        names = db.list()
        if names:
            print(f"Registered faces ({len(names)}):")
            for n in names:
                print(f"  - {n}")
        else:
            print("No faces registered.")
        return

    if args.command == "delete":
        db = FaceDatabase(args.db)
        db.delete(args.name)
        return

    if args.command == "register-device-json":
        payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
        invoke = payload.get("invoke") or payload
        faces = ((invoke.get("data") or {}).get("faces") or [])
        if not faces or "embedding" not in faces[0]:
            print("ERROR: No face embedding found in device JSON")
            sys.exit(1)

        face = faces[0]
        embedding = l2_normalize(np.asarray(face["embedding"], dtype=np.float32))
        db = FaceDatabase(args.db)
        metadata = {
            "score": face.get("score"),
            "quality": face.get("quality"),
            "box": face.get("box"),
            "source_json": os.path.basename(args.json),
            "embedding_source": "device_invoke",
        }
        db.add(args.name, embedding.tolist(), metadata)
        return

    # Resolve models
    if args.command in ("register", "recognize"):
        backend = args.backend
        scrfd_model = args.scrfd_model or str(SCRFD_TFLITE)
        if args.embedding_model:
            emb_model = args.embedding_model
        elif args.float32_embedding and backend == "tflite":
            emb_model = str(MOBILEFACENET_FLOAT32_TFLITE)
        elif backend == "tflite":
            emb_model = str(MOBILEFACENET_INT8_TFLITE)
        else:
            emb_model = str(MOBILEFACENET_ONNX)

        for path in [scrfd_model, emb_model]:
            if not os.path.exists(path):
                print(f"ERROR: Model not found: {path}")
                sys.exit(1)

        pipeline = create_pipeline(backend, scrfd_model, emb_model)

    if args.command == "register":
        result = pipeline.compute(Image.open(args.image), debug=args.debug)
        db = FaceDatabase(args.db)
        metadata = {
            "score": result["score"],
            "quality": result["quality"],
            "pose": result["pose"],
            "source_image": os.path.basename(args.image),
            "scrfd_model": os.path.basename(scrfd_model),
            "embedding_model": os.path.basename(emb_model),
            "backend": backend,
        }
        db.add(args.name, result["embedding"].tolist(), metadata)

    elif args.command == "recognize":
        result = pipeline.compute(Image.open(args.image), debug=args.debug)
        db = FaceDatabase(args.db)
        if not db.list():
            print("ERROR: No faces registered. Register with: python face_db.py register --name <name> <image>")
            sys.exit(1)

        match = db.recognize(result["embedding"], args.threshold)
        print(f"Query confidence: {result['score']:.3f}, quality: {result['quality']:.3f}")
        if match:
            print(f"Match: {match['name']} (similarity={match['similarity']:.4f})")
        else:
            # Show top-3 closest even if below threshold
            scores = []
            for name, entry in db.faces.items():
                stored = np.array(entry["embedding"], dtype=np.float32)
                sim = cosine_similarity(result["embedding"], stored)
                scores.append((name, float(sim)))
            scores.sort(key=lambda x: -x[1])
            print("No match above threshold. Top matches:")
            for name, sim in scores[:3]:
                print(f"  {name}: {sim:.4f}")

    elif args.command == "export":
        db = FaceDatabase(args.db)
        data = db.to_binary()
        with open(args.output, "wb") as f:
            f.write(data)
        print(f"Exported {len(db.list())} faces to {args.output} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
