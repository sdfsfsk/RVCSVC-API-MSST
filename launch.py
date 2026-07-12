import os
import sys
import ctypes
import runpy

script_dir = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(script_dir, "env")

# Native Windows ROCm runtime (same backend used by the RVC WebUI).  Keep the
# old ZLUDA initialization below as a fallback for the legacy Python 3.10 env.
native_rocm = os.environ.get("RVC_NATIVE_ROCM") == "1" or "runtime-rocm" in os.path.normcase(sys.executable)
if native_rocm:
    os.environ["RVC_NATIVE_ROCM"] = "1"
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
    import torch

    print(f"[ROCm] Native PyTorch {torch.__version__}, HIP {torch.version.hip}")
    if not torch.cuda.is_available():
        print("[ERROR] Native ROCm runtime did not detect an AMD GPU")
        sys.exit(1)
    print(f"[ROCm] GPU: {torch.cuda.get_device_name(0)}")

    target_script = sys.argv[1] if len(sys.argv) > 1 else "app_svc.py"
    target_path = os.path.join(script_dir, target_script)
    if not os.path.isfile(target_path):
        print(f"[ERROR] Target script not found: {target_path}")
        sys.exit(1)
    sys.argv = sys.argv[1:]
    runpy.run_path(target_path, run_name="__main__")
    sys.exit(0)

# Conda 环境下，ZLUDA 文件夹是直接放在根目录的，而不是 Lib/site-packages
zluda_dir = os.path.join(env_dir, "zluda")
rocm6_dir = os.path.join(zluda_dir, "rocm6")
rocm5_dir = os.path.join(zluda_dir, "rocm5")

if os.path.isdir(rocm6_dir):
    os.environ["PATH"] = rocm6_dir + ";" + rocm5_dir + ";" + zluda_dir + ";" + env_dir + ";" + os.environ.get("PATH", "")
    sys.path.insert(0, os.path.join(env_dir, "Lib", "site-packages"))
    sys.path.insert(0, zluda_dir)

    nvcuda_path = os.path.join(rocm6_dir, "nvcuda.dll")
    if os.path.exists(nvcuda_path):
        try:
            ctypes.WinDLL(nvcuda_path)
            print("[ZLUDA] Preloaded nvcuda.dll (CUDA->ROCm translation active)")
        except Exception as e:
            print(f"[ZLUDA] Warning: nvcuda.dll preload failed: {e}")

    try:
        import zluda
        print(f"[ZLUDA] ZLUDA runtime loaded (AMD GPU, gfx={getattr(zluda, 'gfx', 'unknown')})")
    except Exception as e:
        print(f"[ZLUDA] ZLUDA load failed: {e}")
else:
    print("[ZLUDA] No ZLUDA directory found, GPU acceleration not available")

import torch

if hasattr(torch, 'version') and torch.version.hip is not None:
    torch.version.hip = None
    print("[ZLUDA] Patched torch.version.hip -> None (bypass amdsmi)")

def _safe_raw_device_count():
    return -1
def _safe_device_count():
    return -1
try:
    import torch.cuda as _cuda_mod
    _cuda_mod._raw_device_count_amdsmi = _safe_raw_device_count
    _cuda_mod._device_count_amdsmi = _safe_device_count
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print(f"[ZLUDA] PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

if __name__ == "__main__":
    target_script = sys.argv[1] if len(sys.argv) > 1 else "app_svc.py"
    target_path = os.path.join(script_dir, target_script)
    if not os.path.isfile(target_path):
        print(f"[ERROR] Target script not found: {target_path}")
        sys.exit(1)
    with open(target_path, encoding="utf-8") as f:
        code = f.read()
    sys.argv = sys.argv[1:]
    exec(compile(code, target_path, "exec"))
