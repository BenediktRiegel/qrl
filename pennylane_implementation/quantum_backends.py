# This file was copied from the project https://github.com/UST-QuAntiL/qhana-plugin-runner in compliance with its license
import pennylane as qml
from enum import Enum
from typing import Optional


class QuantumBackends(Enum):
    # Modified: Removed unused backends
    pennylane_default_qubit = "pennylane_default.qubit"
    pennylane_lightning_qubit = "pennylane_lightning.qubit"
    pennylane_lightning_gpu = "pennylane_lightning.gpu"
    pennylane_lightning_kokkos = "pennylane_lightning.kokkos"   # Modified: Added Kokkos backend

    def get_pennylane_backend(
        self,
        qubit_cnt: int,
        shots: Optional[int],
    ) -> qml.Device:
        """
        Returns the specified quantum backend
        """
        # Modified: Removed unused backends
        if self.name.startswith("pennylane"):
            return qml.device(self.value[10:], wires=qubit_cnt, shots=shots)
        else:
            # TASK_LOGGER.error
            raise NotImplementedError("Unknown pennylane backend specified!")
