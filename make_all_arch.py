import os
import sys
import shutil
import subprocess

all_archs = [
    "rv32i",
    "rv32ic",
    "rv32im",
    "rv32imc",
    "rv32imf",
    "rv32imfc",
    "rv64i",
    "rv64ic",
    "rv64im",
    "rv64imc",
    "rv64imf",
    "rv64imfc"
]

OPT = [
    "",
    "simd"
]

default_flags = [
    "TARGET=vexii_soc",
    "RAM_SIZE=8192K",
    "HEAP_SIZE=4096K",
]

main = "dummy"

def make_arch(arch, opt):
    full_arch = arch if "f" not in arch else arch.replace("f", "af")
    # full_arch += "_zicsr"
    flags = default_flags + [f"MARCH={full_arch}"] + [f"MAIN={main}"] + [f"OPT=\"{opt}\""]
    cmd = ["make"] + flags + ["recompile"]
    print(f"Building architecture: {arch} {opt}")
    cmd = ' '.join(cmd)
    print(f"Command: {cmd}")
    result = subprocess.run([cmd], shell=True)
    if result.returncode != 0:
        print(f"Build failed for architecture: {arch} {opt}")
        sys.exit(1)
    print(f"Build succeeded for architecture: {arch} {opt}\n")
    folder = arch + ("_" + opt if opt != "" else "")
    os.mkdir("elfs/"+ folder) if not os.path.exists("elfs/"+ folder) else None
    shutil.move(f"{main}.elf", f"elfs/{folder}/{main}.elf")

if __name__ == "__main__":
    for arch in all_archs:
        for opt in OPT:
            make_arch(arch, opt)