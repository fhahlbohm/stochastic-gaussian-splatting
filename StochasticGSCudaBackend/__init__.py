from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .StochasticGSCudaBackend.render import render, Settings, TemporalMode, History, create_k_buffer
    __all__ = ['render', 'Settings', 'TemporalMode', 'History', 'create_k_buffer']
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
