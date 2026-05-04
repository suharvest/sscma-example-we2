#!/usr/bin/env python3
"""Download a manageable aligned Glint360K subset from Hugging Face WebDataset shards.

The source dataset is already RetinaFace-aligned to 112x112 and stores identity
labels in sidecar .cls files. This script extracts a bounded subset into a
directory layout that the local ArcFace fine-tuning script can consume:

    datasets/glint360k_subset_112/<identity>/<sample>.jpg

Use this only for research/internal experiments unless the dataset license has
been reviewed for your product.
"""

import argparse
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/gaunernst/glint360k-wds-gz/resolve/main"


def shard_url(index):
    return f"{BASE_URL}/glint360k-{index:04d}.tar.gz"


def open_url_with_retry(url, attempts=5, timeout=120):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = min(60, 2 ** attempt)
            print(f"  retry {attempt}/{attempts} after {delay}s: {exc}", flush=True)
            time.sleep(delay)
    raise last_exc


def parse_cls(data):
    text = data.decode("utf-8", errors="ignore").strip()
    return text.split()[0] if text else None


def write_sample(output_dir, class_id, key, jpg_data, counters, max_images_per_id, eligible_ids=None):
    if class_id is None or jpg_data is None:
        return False
    if eligible_ids is not None and class_id not in eligible_ids:
        return False
    current = counters.get(class_id, 0)
    if max_images_per_id and current >= max_images_per_id:
        return False

    dst_dir = output_dir / str(class_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{key.replace('/', '_')}.jpg"
    if not dst.exists():
        dst.write_bytes(jpg_data)
    counters[class_id] = current + 1
    return True


def scan_shard(url):
    print(f"Scanning {url}", flush=True)
    counts = Counter()
    with open_url_with_retry(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or Path(member.name).suffix.lower() != ".cls":
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                class_id = parse_cls(fileobj.read())
                if class_id is not None:
                    counts[class_id] += 1
    return counts


def extract_shard(url, output_dir, max_images, max_images_per_id, counters, eligible_ids=None):
    print(f"Downloading {url}", flush=True)
    extracted = 0
    pending = {}
    with open_url_with_retry(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".cls"}:
                    continue

                key = str(Path(member.name).with_suffix(""))
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                data = fileobj.read()

                item = pending.setdefault(key, {})
                if suffix == ".cls":
                    item["cls"] = parse_cls(data)
                else:
                    item["jpg"] = data

                if "cls" in item and "jpg" in item:
                    if write_sample(
                        output_dir,
                        item["cls"],
                        key,
                        item["jpg"],
                        counters,
                        max_images_per_id,
                        eligible_ids=eligible_ids,
                    ):
                        extracted += 1
                        if extracted % 1000 == 0:
                            print(f"  extracted {extracted} images from current shard", flush=True)
                    pending.pop(key, None)
                    if max_images and extracted >= max_images:
                        return extracted
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/glint360k_subset_112"))
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=120000)
    parser.add_argument("--max-images-per-id", type=int, default=20)
    parser.add_argument(
        "--min-images-per-id",
        type=int,
        default=0,
        help="Only extract identities that appear at least this many times in the selected shards.",
    )
    parser.add_argument(
        "--balanced-two-pass",
        action="store_true",
        help="First scan selected shards for identity counts, then extract only eligible identities.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counters = {p.name: len(list(p.glob("*.jpg"))) for p in args.output_dir.iterdir() if p.is_dir()}
    total_existing = sum(counters.values())
    print(f"Existing images: {total_existing} in {len(counters)} identities")

    shard_indices = list(range(args.start_shard, args.start_shard + args.num_shards))
    eligible_ids = None
    if args.balanced_two_pass:
        scan_counts = Counter()
        for shard in shard_indices:
            scan_counts.update(scan_shard(shard_url(shard)))
            print(f"Scanned shard {shard:04d}: identities={len(scan_counts)}", flush=True)
        min_images = max(args.min_images_per_id, 1)
        eligible_ids = {class_id for class_id, count in scan_counts.items() if count >= min_images}
        eligible_images = sum(scan_counts[class_id] for class_id in eligible_ids)
        print(
            f"Eligible identities: {len(eligible_ids)} with min_images={min_images}, "
            f"candidate_images={eligible_images}",
            flush=True,
        )

    total_new = 0
    for shard in shard_indices:
        remaining = max(args.max_images - (total_existing + total_new), 0) if args.max_images else 0
        if args.max_images and remaining <= 0:
            break
        try:
            total_new += extract_shard(
                shard_url(shard),
                args.output_dir,
                remaining,
                args.max_images_per_id,
                counters,
                eligible_ids=eligible_ids,
            )
        except Exception as exc:
            print(f"Failed shard {shard:04d}: {exc}", flush=True)
            raise
        print(
            f"Progress: new={total_new}, total={total_existing + total_new}, "
            f"identities={len(counters)}",
            flush=True,
        )

    print(f"Done: new={total_new}, total={total_existing + total_new}, identities={len(counters)}")


if __name__ == "__main__":
    main()
