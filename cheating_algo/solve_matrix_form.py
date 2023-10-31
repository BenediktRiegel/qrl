from typing import List
import time
import torch
# from visualize.rot_swap_value import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
# from visualize.action_value_info import get_action_probs, get_state_values
from logger import Logger


env_matrix = None   # Saves the environmental matrix by get_env_matrix, to avoid multiple computations
terminal_states = {0, 3, 7, 9, 11}  # Determines which states are terminal, i.e. holes or goals
goal_states = {3}   # Determines which states are the goals


def get_env_matrix():
    """
    Returns a matrix env of size |S| x (|S|•|A|) specifying the properties of a 4x4 frozen lake.
    |S|=16 is the size of the state space and |A|=4 is the size of the action space. A state s and an action a are encoded
    into a vector x. This vector is zero everywhere, except at the entry (4*s + a), at this entry its one.
    Computing env @ x gives us a new vector y. The entry k of y is the probability of transitioning to the state y,
    while in state s and executing action a.

    :return: matrix containing the transition properties of the environment
    """
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
    """
    Given the matrix env @ pi, this method returns a vector of the expected rewards one receives, when starting in
    state s and following policy pi.

    :param P: matrix of size |S| x |S|. The entry P[s', s] contains the probability of transitioning to state s' from
    state s under policy pi and the given environment, i.e. P = env @ pi
    :return: returns the reward vector R
    """
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


def get_col_action_probs(policy, s):
    """
    Returns the s'th column of the action matrix pi, given the action parameters
    :param policy: a matrix of size |S| x |A| that contains the probabilities of taking action a while in state s
    :param s: a state, i.e. an integer
    :return: s'th column entry of the action matrix pi
    """
    col = torch.zeros(64, dtype=torch.float64)
    col[s*4:s*4 + 4] = policy[s]
    return col


def get_policy_matrix(policy):
    """
    Returns the policy matrix pi, given the action parameters. The size of pi is |A|•|S| x |S|.
    :param policy: a matrix of size |S| x |A| that contains the probabilities of taking action a while in state s
    :return: policy matrix pi
    """
    matrix = torch.empty((64, 16), dtype=torch.float64)
    for s in range(16):
        matrix[:, s] = get_col_action_probs(policy, s)
    return matrix


def compute_values(I, gamma, P, R):
    """
    Solves v = (I - gamma•P)^(-1) R and returns v. Carefull, here P is (env @ pi)^T.
    :param I: the identity matrix of size |S| x |S|.
    :param gamma: discount factor gamma
    :param P: matrix of size |S| x |S|. The entry P[s, s'] contains the probability of transitioning to state s' from
    state s under policy pi and the given environment, i.e. P = (env @ pi)^T
    :param R: reward vector of size |S|
    :return: returns the reward vector R
    """
    return torch.linalg.solve(I - gamma * P, R)


def get_opt_policy(gamma):
    """
    Does policy itertation to obtain the optimal policy. To calculate the values of a policy, it solves the linear
    system of equations defined by the Bellman equations directly.
    :param gamma: discount factor
    :return: optimal policy and optimal state values
    """
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
        values = torch.linalg.solve(I - gamma * P.T, R)
        # values = torch.linalg.inv(I - gamma * P) @ R
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


def print_det_policy(policy):
    """
    Prints for each state s the action with the highest probability in natural language.
    :param policy: a matrix of size |S| x |A| that contains the probabilities of taking action a while in state s
    """
    a_list = ["right", "down", "left", "up"]
    for s, actions in enumerate(policy):
        print(f"{s}: {a_list[actions.argmax()]}")


def calculate_policy_quality(policy, gamma):
    """
    Given a policy and gamma, this method returns the value of the start state (12'th state)
    :param policy: a matrix of size |S| x |A| that contains the probabilities of taking action a while in state s
    :param gamma: discount factor
    """
    policy = torch.tensor(policy, dtype=torch.float64)
    env = get_env_matrix()
    I = torch.zeros((16, 16), dtype=torch.float64)
    for i in range(16):
        I[i, i] = 1.
    pi = get_policy_matrix(policy)
    P = env @ pi    # transposed transistion matrix
    R = get_reward_vector(P)
    v = compute_values(I, gamma, P.T, R)
    return v[12]



def main():
    """
    Executes get_opt_policy with gamma = 0.8 to retrieve the optimal policy and prints the values, aswell as the policy.
    """
    gamma = 0.8
    policy, v = get_opt_policy(gamma)
    print(f"v: {v}")
    print(f"policy:\n {print_det_policy(policy)}")


if __name__ == "__main__":
    main()
