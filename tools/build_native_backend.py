#!/usr/bin/env python3
"""Compile the web backend modules as stripped native CPython extensions."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


MODULES = ("server", "mocap_receiver", "offline_sync")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    build = args.build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    os.chdir(source)

    compile_args = (
        "-O3",
        "-DNDEBUG",
        "-fvisibility=hidden",
        f"-ffile-prefix-map={source}=web_monitor",
        f"-fmacro-prefix-map={source}=web_monitor",
        f"-ffile-prefix-map={build}=web_monitor",
        f"-fmacro-prefix-map={build}=web_monitor",
    )
    extensions = [
        Extension(
            module,
            [f"{module}.py"],
            extra_compile_args=list(compile_args),
        )
        for module in MODULES
    ]
    extensions = cythonize(
        extensions,
        build_dir=str(build / "generated"),
        compiler_directives={
            "language_level": 3,
            "binding": False,
            "embedsignature": False,
            "emit_code_comments": False,
        },
        annotate=False,
        quiet=True,
    )

    distribution = Distribution({"name": "cvia-native-backend", "ext_modules": extensions})
    command = build_ext(distribution)
    command.build_lib = str(build / "lib")
    command.build_temp = str(build / "objects")
    command.ensure_finalized()
    command.run()

    for module in MODULES:
        matches = list((build / "lib").glob(f"{module}*.so"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one native extension for {module}, found {matches}")
        shutil.copy2(matches[0], output / f"{module}.so")


if __name__ == "__main__":
    main()
