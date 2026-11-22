"""Read-only local environment check. Run: python check_env.py"""
import importlib
import platform
import shutil
import sys
print('Python:', sys.version)
print('Executable:', sys.executable)
print('Platform:', platform.platform())
print('nvidia-smi:', shutil.which('nvidia-smi'))
for name in ['torch', 'torch_geometric', 'numpy', 'scipy']:
    try:
        module = importlib.import_module(name)
        print(name, module.__version__)
        if name == 'torch':
            print('CUDA runtime:', module.version.cuda, 'Available:', module.cuda.is_available())
            if module.cuda.is_available():
                print('GPU:', module.cuda.get_device_name(0))
                print('Compute capability:', module.cuda.get_device_capability(0))
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
