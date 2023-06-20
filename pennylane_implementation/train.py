import time
from visualize.rot_swap_value import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from numpy import cos


def update_plots(fig_path, loss_path, environment, gamma, frames, losses):
    fig = plot_animated_frozen_lake(environment, frames, gamma)
    with open(fig_path, "w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open(loss_path, "w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()


def train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn, loss_fn_params):
    """
    train the model with the given data and parameters
    """

    frames = []
    losses = []

    total_itr = 0

    for i in range(num_iterations):
        start_it_time = time.time()
        for type_itr, (num_sub_itr, itr_type) in enumerate(sub_iterations):
            for sub_i in range(num_sub_itr):
                total_itr += 1
                start_sub_it_time = time.time()

                if itr_type == 1:
                    for parameter in value_qnn.parameters():
                        parameter.requires_grad = False
                elif itr_type == 2:
                    for parameter in action_qnn.parameters():
                        parameter.requires_grad = False

                print(
                    f"Start iterations {i + 1}/{num_iterations}, type_itr: {type_itr + 1}/{len(sub_iterations)}, sub_iteration: {sub_i + 1}/{num_sub_itr}, l_type: {itr_type}")

                # zero gradients
                optimizer.zero_grad(set_to_none=True)

                # calculate loss
                print("Calculate loss")
                if itr_type is None:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn)
                elif itr_type <= 2:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn, l_type=3)
                    losses.append([loss[0].item(), loss[1].item()])
                    loss = loss[itr_type - 1]
                # elif itr_type >= 4:
                else:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn, l_type=itr_type)
                    losses.append([loss.item()])
                print(f"losses: {losses[-1]}")
                print(f"loss: {loss.item()}")

                # backpropagation, adjust weights
                print("Backprop")
                loss.backward()

                print(f"value_qnn.out_q_parameters.grad:\n{value_qnn.out_q_parameters.grad}")

                print("Optimize")
                optimizer.step()

                frames.append(get_frozen_lake_frame(
                    loss_fn_params["environment"], action_qnn, value_qnn,
                    len(loss_fn_params["x_qubits"]), len(loss_fn_params["y_qubits"]),
                    loss_fn_params["gamma"],
                ))

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
                    "Time: {:.4f} min {:.4f} sec with loss: {:.4f} on the training data\n".format(
                        minutes_it, seconds_it, loss.item()
                    )
                )
                if total_itr == 5:
                    total_itr = 0
                    update_plots(loss_fn_params["environment"], loss_fn_params["gamma"], frames, losses)
        # time
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec on the training data\n".format(
                i + 1, num_iterations, minutes_it, seconds_it
            )
        )

    return frames, losses


def train_with_two_opt(
        loss_fn, value_optimizer, action_optimizer, value_scheduler, action_scheduler,
        num_iterations, sub_iterations, action_qnn, value_qnn, loss_fn_params,
        fig_path, loss_path
):
    """
    train the model with the given data and parameters
    """

    frames = []
    losses = []

    total_itr = 0

    value_grads = []

    for i in range(num_iterations):
        start_it_time = time.time()
        for type_itr, (num_sub_itr, itr_type) in enumerate(sub_iterations):
            for sub_i in range(num_sub_itr):
                total_itr += 1
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

                # for parameter in action_qnn.parameters():
                #     parameter.requires_grad = False
                # for parameter in value_qnn.parameters():
                #     parameter.requires_grad = False

                print(
                    f"Start iterations {i + 1}/{num_iterations}, type_itr: {type_itr + 1}/{len(sub_iterations)}, sub_iteration: {sub_i + 1}/{num_sub_itr}, l_type: {itr_type}")

                # calculate loss
                print("Calculate loss")
                if itr_type is None:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn)
                elif itr_type <= 2:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn, l_type=3)
                    losses.append([loss[0].item(), loss[1].item()])
                    loss = loss[itr_type - 1]
                elif itr_type >= 4:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn, l_type=itr_type)
                    losses.append([loss.item()])
                print(f"losses: {losses[-1]}")
                print(f"loss: {loss}")

                # backpropagation, adjust weights
                print("Backprop")
                loss.backward()

                # value_grads.append(value_qnn.in_q_parameters.grad.detach().clone())

                print("Optimize")
                if itr_type == 1:
                    action_optimizer.step()
                elif itr_type == 2:
                    value_optimizer.step()
                else:
                    action_optimizer.step()
                    value_optimizer.step()

                frames.append(get_frozen_lake_frame(
                    loss_fn_params["environment"], action_qnn, value_qnn,
                    len(loss_fn_params["x_qubits"]), len(loss_fn_params["y_qubits"]),
                    loss_fn_params["gamma"],
                    loss_fn_params["end_state_values"],
                ))

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
                    "Time: {:.4f} min {:.4f} sec with loss: {:.4f} on the training data\n".format(
                        minutes_it, seconds_it, loss.item()
                    )
                )
                if total_itr == 5:
                    total_itr = 0
                    update_plots(fig_path, loss_path, loss_fn_params["environment"], loss_fn_params["gamma"], frames, losses)

        # value_scheduler.step()
        # action_scheduler.step()
        # time
        print(value_grads)
        if i >= 3:
            loss_fn_params["backend"].shots = min(loss_fn_params["backend"].shots*2, 10000000)
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec on the training data\n".format(
                i + 1, num_iterations, minutes_it, seconds_it
            )
        )

    return frames, losses
