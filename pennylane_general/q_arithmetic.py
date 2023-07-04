import pennylane as qml
from ccnot import adaptive_ccnot


def cc_increment_register(
    c_wires,
    target_wires,
    ancilla_wires,
    indicator_wire,
    unclean_wires=None,
    indicator_is_zero=True,
):
    """
    Increments a target register by one, if all the control qubits c_wires are 1
    :param c_wires: control qubits
    :param target_wires: target register
    :param ancilla_wires: ancilla qubits
    :param indicator_wire: qubit that indicates, whether the circuit should continue or not
    :param unclean_wires: unclean qubits (their state might not be |0>). They are used for ccnots.
    :param indicator_is_zero: if True, then the indicator_wire is in state |0>, else |1>.
    """
    if indicator_is_zero:
        qml.PauliX((indicator_wire,))  # indicator wire must be 1
    for i in range(len(target_wires) - 1, 0, -1):
        adaptive_ccnot(
            c_wires + [indicator_wire], ancilla_wires, unclean_wires, target_wires[i]
        )  # Increment
        adaptive_ccnot(
            c_wires + target_wires[i:], ancilla_wires, unclean_wires, indicator_wire
        )  # If we had flip from 0->1, then end computation, by setting ancilla wire to 0. Else 1->0 continue computation
        qml.PauliX((target_wires[i]))  # Only negated value of the bit is used later on
    adaptive_ccnot(
        c_wires + [indicator_wire], ancilla_wires, unclean_wires, target_wires[0]
    )  # flip overflow bit, if necessary
    adaptive_ccnot(
        c_wires, ancilla_wires, unclean_wires, indicator_wire
    )  # Reset ancilla wire to one | part 1
    adaptive_ccnot(
        c_wires + target_wires[1:], ancilla_wires, unclean_wires, indicator_wire
    )  # Reset ancilla wire to one | part 2

    for i in range(1, len(target_wires)):
        qml.PauliX((target_wires[i],))  # Reset the negated bits

    if indicator_is_zero:
        qml.PauliX((indicator_wire,))  # reset indicator to input value


def add_registers(
    control_reg, target_reg, indicator_wire, unclean_wires=None, indicator_is_zero=True
):
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))
    for i in range(len(control_reg) - 1, -1, -1):
        cc_increment_register(
            [control_reg[i]],
            target_reg[: i + 1],
            indicator_wire[:-1],
            indicator_wire[-1],
            unclean_wires=control_reg[:i] + control_reg[i + 1 :] + unclean_wires,
            indicator_is_zero=False,
        )
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))


def cc_add_registers(
    controle_wires, control_reg, target_reg, indicator_wire, unclean_wires=None, indicator_is_zero=True
):
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))
    for i in range(len(control_reg) - 1, -1, -1):
        cc_increment_register(
            controle_wires + [control_reg[i]],
            target_reg[: i + 1],
            indicator_wire[:-1],
            indicator_wire[-1],
            unclean_wires=control_reg[:i] + control_reg[i + 1 :] + unclean_wires,
            indicator_is_zero=False,
        )
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))


def add_classical_quantum_registers(
    classical_reg, quantum_reg, indicator_wire, unclean_wires=None, indicator_is_zero=True
):
    if not isinstance(indicator_wire, list):
        indicator_wire = [indicator_wire]
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))
    for i in range(len(classical_reg) - 1, -1, -1):
        if classical_reg[i]:
            cc_increment_register(
                [],
                quantum_reg[: i + 1],
                indicator_wire[:-1],
                indicator_wire[-1],
                unclean_wires=unclean_wires,
                indicator_is_zero=False,
            )
    if indicator_is_zero:
        qml.PauliX((indicator_wire[-1]))


def main():
    reg = list(range(4))
    indicator = list(range(len(reg), len(reg)+1))
    ancilla = list(range(len(reg)+1, len(reg)+2))
    device = qml.device("default.qubit", wires=reg+indicator+ancilla)
    device.shots = 1024

    from debug_utils import snapshots_to_debug_strings

    def circuit():
        qml.PauliX((indicator[0], ))
        qml.Snapshot("Start")
        for i in range(2**len(reg)):
            cc_increment_register([], reg, ancilla, indicator[0], unclean_wires=[], indicator_is_zero=False)
            qml.Snapshot(f"Inc{i}")

        return qml.probs((0, ))

    qnode = qml.QNode(circuit, device)
    snaps = qml.snapshots(qnode)()
    print("\n".join(snapshots_to_debug_strings(
        snaps,
        make_space_at=[reg[0], indicator[0], ancilla[0]],
        show_zero_rounded=False,
    )))
    # print(qml.draw(qnode)())


def main2():
    from wire_utils import get_wires

    num1 = [1, 0]
    num2 = [0, 0]

    for num1 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        for num2 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            wires, total_num_wires = get_wires([len(num1), len(num2), 1, 1])
            reg1, reg2, indicator_wires, ancilla_wires = wires
            device = qml.device("default.qubit", wires=total_num_wires)
            device.shots = 1024

            from debug_utils import snapshots_to_debug_strings

            def circuit():
                for q, b in zip(reg1, num1):
                    if b == 1:
                        qml.PauliX((q,))
                for q, b in zip(reg2, num2):
                    if b == 1:
                        qml.PauliX((q,))

                qml.Snapshot("Start")
                add_registers(reg1, reg2, indicator_wires+ancilla_wires, unclean_wires=[], indicator_is_zero=True)
                # add_classical_quantum_registers(num1, reg2, indicator_wires, unclean_wires=[], indicator_is_zero=True)
                qml.Snapshot(f"Add")

                return qml.probs((0, ))

            qnode = qml.QNode(circuit, device)
            snaps = qml.snapshots(qnode)()
            print("\n".join(snapshots_to_debug_strings(
                snaps,
                make_space_at=[reg2[0], indicator_wires[0], ancilla_wires[0]],
                show_zero_rounded=False,
            )))
            print()
    # print(qml.draw(qnode)())


if __name__ == "__main__":
    main2()
