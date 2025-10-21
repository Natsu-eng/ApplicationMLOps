
import torch
from src.shared.logging import StructuredLogger

logger = StructuredLogger(__name__)
# ======================
# GESTIONNAIRE DE DEVICE
# ======================
class DeviceManager:
    """Gestion dynamique des devices"""
    
    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self._device = None
        self._setup_device()
    
    def _setup_device(self):
        """Configure le device optimal"""
        if self.force_cpu:
            self._device = torch.device("cpu")
            logger.info("Device forcé: CPU")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info(
                "Device sélectionné: CUDA",
                gpu_count=torch.cuda.device_count(),
                gpu_name=torch.cuda.get_device_name(0)
            )
        else:
            self._device = torch.device("cpu")
            logger.info("Device: CPU (CUDA non disponible)")
    
    @property
    def device(self) -> torch.device:
        """Retourne le device actuel"""
        return self._device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Déplace un tensor sur le device"""
        return tensor.to(self._device)