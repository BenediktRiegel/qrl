from typing import List
import time
import torch
# from visualize.rot_swap_value import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
# from visualize.action_value_info import get_action_probs, get_state_values
from logger import Logger


env_matrix = None
terminal_states = {0, 3, 7, 9, 11}
goal_states = {3}


def get_env_matrix():
    global env_matrix
    if env_matrix is None:
        global terminal_states
        env_matrix = torch.zeros((16, 64), dtype=torch.float64)
        m_entry = 0
        for s in range(16):
            for a in range(4):
                next_s_prob = torch.zeros(16)
                if s in terminal_states:
                    next_s_prob[s] = 3.
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
                            next_s_prob[next_s] += prob
                env_matrix[:, m_entry] = next_s_prob
                m_entry += 1
        env_matrix /= 3.

    return env_matrix


def get_reward_vector(P):
    R = torch.zeros(16, dtype=torch.float64)
    for s in range(16):
        s_vec = torch.zeros(16, dtype=torch.float64)
        s_vec[s] = 1
        trans_probs = P @ s_vec
        for s_next in range(16):
            if s_next not in goal_states:
                trans_probs[s_next] = 0
        r = trans_probs.sum()
        R[s] = r
    return R


def get_action_probs(action_params, s):
    # p = action_params[s]
    # probs = torch.tensor([torch.cos((p[0] + p[2])/2.), torch.sin((p[0] + p[2])/2.)])
    # probs = torch.kron(probs, torch.tensor([torch.cos((p[1] + p[2])/2.), torch.sin((p[1] + p[2])/2.)]))
    return action_params[s]


def get_col_action_probs(action_params, s):
    col = torch.zeros(64, dtype=torch.float64)
    col[s*4:s*4 + 4] = get_action_probs(action_params, s)
    return col


def get_policy_matrix(action_params):
    matrix = torch.empty((64, 16), dtype=torch.float64)
    for s in range(16):
        matrix[:, s] = get_col_action_probs(action_params, s)
    return matrix


def get_value_vector(value_params):
    return torch.cos(value_params / 2.)


def value_loss(P, R, v, gamma):
    # v_max = r_max / (1 - gamma) --r_max=1--> v_max = 1/(1 - gamma) -> R / v_max = R * (1 - gamma)
    loss = torch.square(v - (1-gamma)*R - gamma * (P @ v))
    return loss.sum() / 16 / 3 / ((1-gamma)**2) / (2 + gamma ** 2)


def sample_value_loss(P, R, v, gamma, shots):
    one_prob = 0.5 - value_loss(P, R, v, gamma) / 2.
    samples = torch.floor(torch.rand(shots) + one_prob)
    sampled_loss = ((-2 * samples) + 1).sum() / shots
    return sampled_loss


def compute_value_grad(value_params, P, R, gamma, shots):
    value_params.grad = torch.zeros(value_params.shape, dtype=value_params.dtype)
    shift = torch.pi / 4.
    for idx, p in enumerate(value_params):
        value_params[idx] = p + shift
        loss_plus = sample_value_loss(P, R, get_value_vector(value_params), gamma, shots)
        value_params[idx] = p - shift
        loss_minus = sample_value_loss(P, R, get_value_vector(value_params), gamma, shots)
        value_params.grad[idx] = loss_plus - loss_minus
        value_params[idx] = p


def compute_values(I, gamma, P, R):
    return torch.linalg.inv(I - gamma * P) @ R


def get_opt_policy(gamma):
    env = get_env_matrix()
    I = torch.zeros((16, 16), dtype=torch.float64)
    for i in range(16):
        I[i, i] = 1.
    # All going right
    policy = torch.zeros((16, 4))
    policy[:, 0] += 1

    while True:
        old_policy = policy.clone()
        # Calc policy
        P = env @ get_policy_matrix(old_policy)
        R = get_reward_vector(P)
        values = torch.linalg.inv(I - gamma * P) @ R
        # Update policy
        policy = torch.zeros((16, 4))
        for s, v in enumerate(values):
            # Evaluate actions
            best_a = torch.tensor([env[:, s*4 + a] @ values for a in range(4)]).argmax()
            policy[s, best_a] = 1.

        print(f"Difference: {torch.abs(old_policy - policy).sum()}")
        if torch.abs(old_policy - policy).sum() < 1e-10:
            break

    return policy, values


def get_next_s(policy, s):
    env = get_env_matrix()
    policy = get_policy_matrix(policy)
    P = env @ policy
    probs = P[:, s]
    print({next_s: prob for next_s, prob in enumerate(probs) if prob != 0})


def main():
    gamma = 0.8
    policy, v = get_opt_policy(gamma)
    print(f"v: {v}")
    env = get_env_matrix()
    I = torch.zeros((16, 16), dtype=torch.float64)
    for i in range(16):
        I[i, i] = 1.
    P = env @ get_policy_matrix(policy)
    R = get_reward_vector(P)
    print(f"R: {R}")
    v2 = compute_values(I, gamma, env @ get_policy_matrix(policy), R)
    print(f"v2: {v2}")
    print(f"v - v2: {v - v2}")
    get_next_s(policy, 7)
    # policy = torch.tensor([
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    # ])
    # env = get_env_matrix()
    # I = torch.zeros((16, 16), dtype=torch.float64)
    # for i in range(16):
    #     I[i, i] = 1.
    # P = env @ get_policy_matrix(policy)
    # R = get_reward_vector(P)
    # print(f"v1: {compute_values(I, gamma, P, R)}")


if __name__ == "__main__":
    main()
