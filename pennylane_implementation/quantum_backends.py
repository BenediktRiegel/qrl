import pennylane as qml
from enum import Enum
from typing import Optional
# from qiskit import IBMQ


class QuantumBackends(Enum):
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_santiago = "ibmq_santiago"
    ibmq_manila = "ibmq_manila"
    ibmq_bogota = "ibmq_bogota"
    ibmq_quito = "ibmq_quito"
    ibmq_belem = "ibmq_belem"
    ibmq_lima = "ibmq_lima"
    ibmq_armonk = "ibmq_armonk"
    pennylane_default_qubit = "pennylane_default.qubit"
    pennylane_lightning_qubit = "pennylane_lightning.qubit"
    pennylane_lightning_gpu = "pennylane_lightning.gpu"
    pennylane_lightning_kokkos = "pennylane_lightning.kokkos"

    def get_pennylane_backend(
        self,
        ibmq_token: str,
        custom_backend_name: str,
        qubit_cnt: int,
        shots: Optional[int],
    ) -> qml.Device:
        if self.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.name[4:]
            # return qml.device(
            #     "qiskit.aer", wires=qubit_cnt, backend=aer_backend_name, shots=shots
            # )
        elif self.name.startswith("ibmq"):
            pass
            # Use IBMQ backend
            # provider = IBMQ.enable_account(ibmq_token)

            # return qml.device(
            #     "qiskit.ibmq",
            #     wires=qubit_cnt,
            #     backend=self.name,
            #     provider=provider,
            #     shots=shots,
            # )
        elif self.name.startswith("custom_ibmq"):
            pass
            # Use custom IBMQ backend
            # provider = IBMQ.enable_account(ibmq_token)

            # return qml.device(
            #     "qiskit.ibmq",
            #     wires=qubit_cnt,
            #     backend=custom_backend_name,
            #     provider=provider,
            #     shots=shots,
            # )
        elif self.name.startswith("pennylane"):
            return qml.device(self.value[10:], wires=qubit_cnt, shots=shots)
        else:
            # TASK_LOGGER.error
            raise NotImplementedError("Unknown pennylane backend specified!")
