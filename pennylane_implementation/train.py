import time
from visualize import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss


def train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn, loss_fn_params):
    """
    train the model with the given data and parameters
    """

    frames = []
    losses = []

    for i in range(num_iterations):
        start_it_time = time.time()
        for type_itr, (num_sub_itr, itr_type) in enumerate(sub_iterations):
            for sub_i in range(num_sub_itr):
                start_sub_it_time = time.time()

                if itr_type == 1:
                    for parameter in value_qnn.parameters():
                        parameter.requires_grad = False
                elif itr_type == 2:
                    for parameter in action_qnn.parameters():
                        parameter.requires_grad = False

                print(f"Start iterations {i+1}/{num_iterations}, type_itr: {type_itr+1}/{len(sub_iterations)}, sub_iteration: {sub_i+1}/{num_sub_itr}, l_type: {itr_type}")

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
                elif itr_type >= 4:
                    loss = loss_fn(**loss_fn_params, action_qnn=action_qnn, value_qnn=value_qnn, l_type=itr_type)
                    losses.append([loss.item()])
                print(f"loss: {loss}")

                # backpropagation, adjust weights
                print("Backprop")
                loss.backward()

                print("Optimize")
                optimizer.step()

                frames.append(get_frozen_lake_frame(
                    loss_fn_params["environment"], action_qnn, value_qnn,
                    len(loss_fn_params["x_qubits"]), len(loss_fn_params["y_qubits"])
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
        # time
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec on the training data\n".format(
                i + 1, num_iterations, minutes_it, seconds_it
            )
        )

        fig = plot_animated_frozen_lake(loss_fn_params["environment"], frames)
        with open("plots/fig.html", "w") as f:
            f.write(fig.to_html())
            f.close()

        fig = plot_loss(losses)
        with open("plots/loss.html", "w") as f:
            f.write(fig.to_html())
            f.close()
    return frames, losses
