import time
import torch
# from visualize.rot_swap_value import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
# from visualize.action_value_info import get_action_probs, get_state_values
from logger import Logger
from numpy import sqrt
from struct import pack, unpack
from binascii import hexlify


trans_model = None
# terminal_states = {5, 7, 11, 12, 15}
# goal_states = {15}
terminal_states = {0, 3, 7, 9, 11}
hole_states = {}
# goal_states = {}
# hole_states = {0, 7, 9, 11}
goal_states = {3}
v_max = None


def get_v_max(gamma, end_state_values):
    global v_max
    if v_max is None:
        v_max = (1. / (1-gamma)) if end_state_values else 1.
    return v_max

def get_trans_model():
    global trans_model
    if trans_model is None:
        global terminal_states
        trans_model = []
        for s in range(16):
            trans_model.append([dict(), dict(), dict(), dict()])
            for a in range(4):
                if s in terminal_states:
                    trans_model[s][a][s] = 3.
                else:
                    slip_a_probs = [1, 1, 0, 1]
                    for slip_a, prob in enumerate(slip_a_probs):
                        if prob != 0:
                            slip_a = (slip_a + a) % 4
                            next_s = s
                            if slip_a == 0:
                                if (s % 4) != 3:
                                    next_s += 1
                            elif slip_a == 1:
                                if s > 3:
                                    next_s -= 4
                            elif slip_a == 2:
                                if (s % 4) != 0:
                                    next_s -= 1
                            elif slip_a == 3:
                                if s < 12:
                                    next_s += 4
                            if next_s not in trans_model[s][a]:
                                trans_model[s][a][next_s] = 0
                            trans_model[s][a][next_s] += prob

        for k1, c1 in enumerate(trans_model):
            for k2, c2 in enumerate(c1):
                for k3, c3 in c2.items():
                    trans_model[k1][k2][k3] = c3 / 3.

    return trans_model


def get_action_probs(action_params, s, eps: float = 0.0):
    p = action_params[s]
    probs = torch.square(torch.tensor([torch.cos((p[0] + p[2])/2.), torch.sin((p[0] + p[2])/2.)]))
    probs = torch.kron(probs, torch.square(torch.tensor([torch.cos((p[1] + p[2])/2.), torch.sin((p[1] + p[2])/2.)])))
    probs = (probs * (1-eps)) + eps
    return probs


def get_policy(action_params, eps: float = 0.0):
    return [get_action_probs(action_params, s, eps) for s in range(16)]


def get_values(value_params, end_state_values):
    values = torch.cos(value_params / 2.)
    if not end_state_values:
        values[list(terminal_states)] = 0
    return values


def value_loss(policy, trans_model, v, next_v, gamma, end_state_values, without_factors=False):
    # v_max = r_max / (1 - gamma) --r_max=1--> v_max = 1/(1 - gamma) -> R / v_max = R * (1 - gamma)
    global goal_states
    v_max = get_v_max(gamma, end_state_values)
    loss = 0
    next_v = gamma * next_v
    for s, (action_probs, s_value) in enumerate(zip(policy, v)):
        if s not in terminal_states or end_state_values:
            for a, a_prob in enumerate(action_probs):
                for next_s, trans_prob in trans_model[s][a].items():
                    r = 1./v_max if next_s in goal_states else 0
                    r = -1./v_max if next_s in hole_states else r
                    loss += a_prob * trans_prob * torch.square(s_value - r - next_v[next_s])
    loss /= 16
    if not without_factors:
        loss = loss / 3 / (2 + gamma**2)
    return loss


def sample_value_loss(policy, trans_model, v, next_v, gamma, end_state_values, shots, qpe_qubits: int, max_qpe_prob: float):
    v_max = get_v_max(gamma, end_state_values)
    if shots is None:
        return value_loss(policy, trans_model, v, next_v, gamma, end_state_values, without_factors=True) * (v_max**2)
    # Retrieve "exact" probability of the loss qubit being in state |1>
    one_prob = 0.5 - value_loss(policy, trans_model, v, next_v, gamma, end_state_values) / 2.
    if qpe_qubits and qpe_qubits > 0:
        # Get shots many result from qpe
        results = qpe_one_prob(one_prob, qpe_qubits, shots, max_qpe_prob)
        # Get Median. Usually, we would have to use the Median of Median algorithm, to achieve O(shots)
        results.sort()
        one_prob = results[len(results) // 2]
        return (1 - 2*one_prob) * 3 * (v_max**2) * (2 + gamma ** 2)
    # sample normally
    samples = torch.floor(torch.rand(shots) + one_prob)
    sampled_loss = ((-2 * samples) + 1).sum() / shots
    return sampled_loss * 3 * (v_max**2) * (2 + gamma ** 2)


def compute_value_grad(value_params, policy, next_values, trans_model, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob):
    value_params.grad = torch.zeros(value_params.shape, dtype=value_params.dtype)
    shift = torch.pi / 4.
    for idx, p in enumerate(value_params):
        p = p.clone()
        value_params[idx] = p + shift
        loss_plus = sample_value_loss(policy, trans_model, get_values(value_params, end_state_values), next_values, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
        value_params[idx] = p - shift
        loss_minus = sample_value_loss(policy, trans_model, get_values(value_params, end_state_values), next_values, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
        value_params.grad[idx] = loss_plus - loss_minus
        value_params[idx] = p


def action_loss(policy, trans_model, v, gamma, end_state_values, without_factors=False):
    # v_max = r_max / (1 - gamma) --r_max=1--> v_max = 1/(1 - gamma) -> R / v_max = R * (1 - gamma)
    global goal_states
    v_max = get_v_max(gamma, end_state_values)
    loss = 0
    next_v = gamma * v.clone().detach()
    for s, action_probs in enumerate(policy):
        if s not in terminal_states or end_state_values:
            for a, a_prob in enumerate(action_probs):
                for next_s, trans_prob in trans_model[s][a].items():
                    r = 1./v_max if next_s in goal_states else 0
                    r = -1./v_max if next_s in hole_states else r
                    loss -= a_prob * trans_prob * (r + next_v[next_s])
    loss /= 16
    if not without_factors:
        loss = loss / sqrt(2) / sqrt(1 + gamma ** 2)

    return loss


def sample_action_loss(policy, trans_model, v, gamma, end_state_values, shots: int, qpe_qubits: int, max_qpe_prob: float):
    v_max = get_v_max(gamma, end_state_values)
    if shots is None:
        return action_loss(policy, trans_model, v, gamma, end_state_values, without_factors=True) * v_max
    # Retrieve "exact" probability of the loss qubit being in state |1>
    one_prob = 0.5 - action_loss(policy, trans_model, v, gamma, end_state_values) / 2.
    if qpe_qubits and qpe_qubits > 0:
        # Get shots many result from qpe
        results = qpe_one_prob(one_prob, qpe_qubits, shots, max_qpe_prob)
        # Get Median. Usually, we would have to use the Median of Median algorithm, to achieve O(shots)
        results.sort()
        one_prob = results[len(results) // 2 + 1]
        return (1 - 2*one_prob) * sqrt(2) * v_max * sqrt(1 + gamma ** 2)
    # Sample normally
    samples = torch.floor(torch.rand(shots) + one_prob)
    sampled_loss = ((-2 * samples) + 1).sum() / shots
    return sampled_loss * sqrt(2) * v_max * sqrt(1 + gamma ** 2)


def compute_action_grad(action_params, v, trans_model, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob):
    action_params.grad = torch.zeros(action_params.shape, dtype=action_params.dtype)
    shift = torch.pi / 4.
    for s, state_p in enumerate(action_params):
        for idx, p in enumerate(state_p):
            p = p.clone()
            action_params[s, idx] = p + shift
            loss_plus = sample_action_loss(get_policy(action_params), trans_model, v, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
            action_params[s, idx] = p - shift
            loss_minus = sample_action_loss(get_policy(action_params), trans_model, v, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
            action_params.grad[s, idx] = loss_plus - loss_minus
            action_params[s, idx] = p


def get_qpe_output_prob(delta, l, N):
    exp1 = N*delta - l
    exp2 = delta - l/N
    if exp2 == 0:
        return 1.0
    return torch.square(torch.abs((1-torch.exp(2j*torch.pi*exp1)) / (1-torch.exp(2j*torch.pi*exp2)))) / N / N


def best_phi_approximation(phi: torch.tensor, num_bits: int):
    phi = phi.clone().detach()
    b_bits = torch.zeros(num_bits)
    for bit in range(1, num_bits+1):
        phi *= 2
        if phi >= 1.0:
            b_bits[bit-1] = 1
            phi -= 1
        if phi <= 0:
            break
    small_b = torch.tensor(0., dtype=phi.dtype)
    # big_b = torch.tensor(0., dtype=phi.dtype)
    for little_end_bit, big_end_bit in zip(reversed(b_bits), b_bits):
        small_b /= 2.
        small_b += (little_end_bit/2.0)
        # big_b *= 2.
        # big_b += big_end_bit

    return small_b  # , big_b


def sample_qpe(phi: torch.tensor, num_qubits: int, shots: int, max_prob: float):
    N = 2**num_qubits
    b = best_phi_approximation(phi, num_qubits)   # best phi approximation
    delta = phi - b
    if delta < 0:
        # return torch.zeros(shots) + (b / N)
        raise ValueError(f"delta must be greater or equal to 0, but got {phi} - {b} = {delta}")
    # print(f"phi-b/N = {phi - b/N}")
    threshold = torch.rand(shots)
    result = torch.empty(shots)
    in_progress = torch.ones(shots, dtype=torch.bool)
    current_prob = 0
    num_itr = 0
    for l1, l2 in zip(range(N // 2), range(1, N // 2 + 1)):
        current_prob += get_qpe_output_prob(delta, l1, num_qubits)
        new_angle = b + l1 / N
        if new_angle > 1:
            new_angle -= 1
        indices_in_progress = torch.where(in_progress)
        chosen = torch.floor(current_prob + (1 - threshold[indices_in_progress]))
        result[indices_in_progress] = chosen * new_angle
        in_progress[indices_in_progress] = (chosen == 0)

        current_prob += get_qpe_output_prob(delta, l2, num_qubits)
        new_angle = b - l2 / N
        if new_angle < 0:
            new_angle += 1
        indices_in_progress = torch.where(in_progress)
        chosen = torch.floor(current_prob + (1 - threshold[indices_in_progress]))
        result[indices_in_progress] = chosen * new_angle
        in_progress[indices_in_progress] = (chosen == 0)

        num_itr += 1
        if not torch.any(in_progress):
            break
        if current_prob > max_prob:
            break
    # Make in_progress the worst case
    worst_value = 0.5 if (phi < 0.25 or phi > 0.75) else 0.
    indices_in_progress = torch.where(in_progress)
    result[indices_in_progress] = worst_value
    # print(f"qpe took {num_itr} iterations")
    return result


def qpe_one_prob(one_prob, num_qubits, shots, max_qpe_prob: float):
    theta = torch.arcsin(torch.sqrt(one_prob)) / torch.pi
    qpe_theta = sample_qpe(theta, num_qubits, shots, max_qpe_prob)
    # print(f"theta: {theta}, qpe_theta: {qpe_theta}")
    new_one_prob = torch.square(torch.sin(qpe_theta*torch.pi))
    return new_one_prob


def train(
        value_optimizer, action_optimizer, num_iterations, sub_iterations,
        action_params, value_params, gamma, eps, end_state_values, shots, qpe_qubits, max_qpe_prob,
        logger: Logger
):
    """
    train the model with the given data and parameters
    """
    trans_model = get_trans_model()
    value_params.grad = torch.zeros(value_params.shape, dtype=value_params.dtype)
    action_params.grad = torch.zeros(action_params.shape, dtype=action_params.dtype)

    action_probs = [entry.tolist() for entry in get_policy(action_params)]
    state_values = (get_values(value_params, end_state_values) * get_v_max(gamma, end_state_values)).tolist()
    logger.log(0, 0, 0, 0, 0, 0, 0, action_probs, state_values, value_params.grad.tolist(), action_params.grad.tolist(), 0, 0)

    total_start = time.time()
    for i in range(num_iterations):
        start_it_time = time.time()
        for type_itr, (min_change, num_sub_itr, itr_type) in enumerate(sub_iterations):
            for sub_i in range(num_sub_itr):
                start_sub_it_time = time.time()

                print(f"Start iterations {i + 1}/{num_iterations}, type_itr: {type_itr + 1}/{len(sub_iterations)}, sub_iteration: {sub_i + 1}/{num_sub_itr}, l_type: {itr_type}")

                print("copy parameters")
                old_action_params = action_params.clone().detach()
                old_value_params = value_params.clone().detach()

                # calculate loss
                print("Calculate loss")
                policy = get_policy(action_params, eps)
                values = get_values(value_params, end_state_values)
                next_values = values.clone().detach()
                action_loss = sample_action_loss(policy, trans_model, next_values, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
                value_loss = sample_value_loss(policy, trans_model, values, next_values, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
                print(f"value_loss: {value_loss}, action_loss: {action_loss}")

                # backpropagation, adjust weights
                print("Compute gradient")
                if itr_type == 1:
                    action_optimizer.zero_grad()
                    compute_action_grad(action_params, values, trans_model, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
                    # action_loss.backward()
                else:
                    value_optimizer.zero_grad()
                    compute_value_grad(value_params, policy, next_values, trans_model, gamma, end_state_values, shots, qpe_qubits, max_qpe_prob)
                    # value_loss.backward()

                # print(f"action_grads: {action_params.grad}")
                # print(f"value_grads: {value_params.grad}")
                # print(f"action grads: {action_qnn.in_q_parameters.grad}")
                # value_grads.append(value_qnn.in_q_parameters.grad.detach().clone())
                v_grads = torch.zeros(value_params.shape) if value_params.grad is None else value_params.grad.clone()
                a_grads = torch.zeros(action_params.shape) if action_params.grad is None else action_params.grad.clone()

                print("Optimize")
                value_params_change = 0
                action_params_change = 0
                insufficient_change = False
                if itr_type == 1:
                    action_optimizer.step()
                    action_params_change = torch.max(torch.abs(action_params - old_action_params)).item()
                    insufficient_change = action_params_change < min_change
                    # print(f"action diff: {torch.linalg.norm(action_params - clone)}")
                else:
                    value_optimizer.step()
                    value_params_change = torch.max(torch.abs(value_params - old_value_params)).item()
                    insufficient_change = value_params_change < min_change
                    # print(f"value diff: {torch.linalg.norm(value_params - clone)}")

                # time
                total_sub_it_time = time.time() - start_sub_it_time
                minutes_it = total_sub_it_time // 60
                seconds_it = round(total_sub_it_time - minutes_it * 60)

                print(
                    "Time: {:.4f} min {:.4f} sec with on the training data\n".format(
                        minutes_it, seconds_it
                    )
                )
                action_probs = [entry.tolist() for entry in get_policy(action_params)]
                state_values = (get_values(value_params, end_state_values) * get_v_max(gamma, end_state_values)).tolist()
                # v_grads = torch.zeros(value_params.shape) if value_params.grad is None else value_params.grad
                # a_grads = torch.zeros(action_params.shape) if action_params.grad is None else action_params.grad
                logger.log(i+1, type_itr, sub_i, total_sub_it_time, time.time() - total_start, value_loss.item(), action_loss.item(), action_probs, state_values, v_grads.tolist(), a_grads.tolist(), action_params_change, value_params_change)

                if insufficient_change:
                    break

        # time
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec on the training data\n".format(
                i + 1, num_iterations, minutes_it, seconds_it
            )
        )

    total_it_time = time.time() - total_start
    minutes_it = total_it_time // 60
    seconds_it = round(total_it_time - minutes_it * 60)
    print(f"Total training time: {minutes_it} min {seconds_it} sec")


if __name__ == "__main__":
    phi = torch.tensor(0.23895668439977433, dtype=torch.float64)
    # ValueError: delta must be greater or equal to 0, but got 0.23895668439977433 - 0.23895668983459473 = -5.434820399274898e-09
    small_b, big_b = best_phi_approximation(phi, 32)
    delta = phi - small_b
    print(f"{phi} - {small_b} = {delta}")
    scaled_b = small_b*(2**32)
    print(f"{scaled_b} - {big_b} = {scaled_b-big_b}")
