from typing import List
import time
from visualize.rot_swap_value import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from visualize.action_value_info import get_action_probs, get_state_values
from logger import Logger


def update_plots(fig_path, loss_path, environment, gamma, frames, losses):
    fig = plot_animated_frozen_lake(environment, frames, gamma)
    with fig_path.open("w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with loss_path.open("w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()


def train(
        loss_fn, value_optimizer, action_optimizer, value_scheduler, action_scheduler,
        num_iterations, sub_iterations, action_qnn, value_qnn, loss_fn_params,
        fig_path, loss_path, logger: Logger, states_to_log: List[List[int]]
):
    """
    train the model with the given data and parameters
    """

    frames = []
    losses = []

    total_itr = 0

    value_grads = []

    total_start = time.time()
    for i in range(num_iterations):
        start_it_time = time.time()
        for type_itr, (num_sub_itr, itr_type) in enumerate(sub_iterations):
            itr_type = 1 if (itr_type > 2 or itr_type < 1 or not itr_type) else itr_type
            for sub_i in range(num_sub_itr):
                # total_itr += 1
                start_sub_it_time = time.time()

                if itr_type == 1:
                    for parameter in value_qnn.parameters():
                        parameter.requires_grad = False
                    # zero gradients
                    action_optimizer.zero_grad(set_to_none=True)
                elif itr_type == 2:
                    for parameter in action_qnn.parameters():
                        parameter.requires_grad = False
                    # zero gradients
                    value_optimizer.zero_grad(set_to_none=True)

                print(
                    f"Start iterations {i + 1}/{num_iterations}, type_itr: {type_itr + 1}/{len(sub_iterations)}, sub_iteration: {sub_i + 1}/{num_sub_itr}, l_type: {itr_type}")

                # calculate loss
                print("Calculate loss")
                loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn)
                losses.append([loss[0].item(), loss[1].item()])
                loss = loss[itr_type - 1]
                print(f"losses: {losses[-1]}")
                print(f"loss: {loss}")

                # backpropagation, adjust weights
                print("Backprop")
                loss.backward()

                # print(f"action grads: {action_qnn.in_q_parameters.grad}")
                # value_grads.append(value_qnn.in_q_parameters.grad.detach().clone())

                print("Optimize")
                if itr_type == 1:
                    action_optimizer.step()
                elif itr_type == 2:
                    value_optimizer.step()
                else:
                    action_optimizer.step()
                    value_optimizer.step()

                # frames.append(get_frozen_lake_frame(
                #     loss_fn_params["environment"], action_qnn, value_qnn,
                #     len(loss_fn_params["x_qubits"]), len(loss_fn_params["y_qubits"]),
                #     loss_fn_params["gamma"],
                #     loss_fn_params["end_state_values"],
                # ))

                if itr_type == 1:
                    for parameter in value_qnn.parameters():
                        parameter.requires_grad = True
                elif itr_type == 2:
                    for parameter in action_qnn.parameters():
                        parameter.requires_grad = True

                # time
                total_sub_it_time = time.time() - start_sub_it_time
                minutes_it = total_sub_it_time // 60
                seconds_it = round(total_sub_it_time - minutes_it * 60)

                print(
                    "Time: {:.4f} min {:.4f} sec with on the training data\n".format(
                        minutes_it, seconds_it
                    )
                )
                log_v_loss = losses[-1]
                log_a_loss = 0
                if isinstance(log_v_loss, list):
                    log_v_loss = losses[-1][0]
                    log_a_loss = losses[-1][1]
                action_probs = get_action_probs(states_to_log, action_qnn, loss_fn_params["action_qubits"])
                state_values = get_state_values(states_to_log, value_qnn, loss_fn_params["environment"].r_m / (1 - loss_fn_params["gamma"]))
                logger.log(i, type_itr, sub_i, total_sub_it_time, time.time() - total_start, log_v_loss, log_a_loss, action_probs, state_values)
                # if total_itr == 5:
                #     total_itr = 0
                #     update_plots(fig_path, loss_path, loss_fn_params["environment"], loss_fn_params["gamma"], frames, losses)

        # value_scheduler.step()
        # action_scheduler.step()
        # time
        print(value_grads)
        # if i >= 3:
        #     loss_fn_params["backend"].shots = min(loss_fn_params["backend"].shots*2, 10000000)
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec on the training data\n".format(
                i + 1, num_iterations, minutes_it, seconds_it
            )
        )

    return frames, losses
