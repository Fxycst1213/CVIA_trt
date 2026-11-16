"""Install a pinned official Linux MediaMTX binary inside this project; no sudo needed."""
import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

VERSION = "1.21.3"
PACKAGES = {
    "aarch64": ("arm64", "27f038237cb5f77a724f6afc58eca2798f2bc4edf9989895631a80eeb41a4fe2"),
    "x86_64": ("amd64", "0a2ea7eba96023631efa1bea7804823a9145b043b6646a8a05798d4ac24ed635"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional official archive for offline installation")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent / "bin")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() not in PACKAGES:
        parser.error("This installer supports Linux ARM64 and AMD64")
    destination = args.destination.resolve()
    if (destination / "mediamtx").is_file():
        print(f"MediaMTX already installed: {destination / 'mediamtx'}")
        return 0
    arch, checksum = PACKAGES[platform.machine()]
    name = f"mediamtx_v{VERSION}_linux_{arch}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".mediamtx-", dir=destination) as temporary:
            temporary = Path(temporary)
            archive = args.archive
            if archive is None:
                archive = temporary / name
                url = f"https://github.com/bluenviron/mediamtx/releases/download/v{VERSION}/{name}"
                print(f"Downloading official MediaMTX v{VERSION} ({arch})", flush=True)
                with urlopen(url, timeout=30) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
            digest = hashlib.sha256()
            with archive.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != checksum:
                raise ValueError("Official archive SHA256 mismatch; nothing was installed")
            with tarfile.open(archive) as bundle:
                for member_name, output_name in (("mediamtx", "mediamtx"), ("LICENSE", "MEDIAMTX_LICENSE")):
                    member = bundle.getmember(member_name)
                    if not member.isfile():
                        raise ValueError(f"Unexpected archive entry: {member_name}")
                    with bundle.extractfile(member) as source, (temporary / output_name).open("wb") as output:
                        shutil.copyfileobj(source, output)
            (temporary / "mediamtx").chmod(0o755)
            (temporary / "MEDIAMTX_LICENSE").replace(destination / "MEDIAMTX_LICENSE")
            (temporary / "mediamtx").replace(destination / "mediamtx")
    except (OSError, ValueError, tarfile.TarError, KeyError) as error:
        print(f"MediaMTX installation failed: {error}", file=sys.stderr)
        return 1
    print(f"Installed MediaMTX v{VERSION}: {destination / 'mediamtx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
