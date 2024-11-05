import json
import matplotlib.pyplot as plt


def plot(data, directory):
    # print("Plotting", start_time, "length", len(data["epoch"]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(data["epoch"], data["train_loss"], label="train_loss")
    ax[0].plot(data["epoch"], data["val_loss"], label="val_loss")
    ax[0].plot(data["epoch"], data["bnll_loss"], label="bnll_loss", linestyle="--")
    ax[0].plot(data["epoch"], data["recon_loss"], label="recon_loss", linestyle="--")
    ax[0].plot(
        data["epoch"], data["smoothness_loss"], label="smoothness_loss", linestyle="--"
    )
    ax[0].scatter(
        data["saved"],
        [data["val_loss"][x - 1] for x in data["saved"]],
        label="saved",
        color="red",
    )

    ax0_2 = ax[0].twinx()

    lr = [x * 1e6 for x in data["lr"]]
    ax0_2.plot(data["epoch"], lr, label="lr * 1e6", color="green")

    colors = ["red", "blue", "green", "brown"]
    for i in range(len(data["alpha"][0])):
        ax[1].plot(
            data["epoch"],
            [x[i] for x in data["alpha"]],
            label=f"alpha_{i}",
            color=colors[i],
            linestyle="-",
        )
        ax[1].plot(
            data["epoch"],
            [x[i] for x in data["beta"]],
            label=f"beta_{i}",
            color=colors[i],
            linestyle="--",
        )
        ax[1].plot(
            data["epoch"],
            [x[i] for x in data["weights"]],
            label=f"weight_{i}",
            color=colors[i],
            linestyle=":",
        )

    ax[2].plot(
        data["epoch"],
        data["bnll_loss_grad_magnitude"],
        label="bnll_gradients",
        color="blue",
    )
    ax[2].plot(
        data["epoch"],
        data["recon_loss_grad_magnitude"],
        label="recon_gradients",
        color="red",
    )
    ax[2].plot(
        data["epoch"],
        data["smoothness_loss_grad_magnitude"],
        label="smoothness_gradients",
        color="green",
    )

    ax2_2 = ax[2].twinx()
    ax2_2.plot(
        data["epoch"],
        data["bnll_weight"],
        label="bnll_weight",
        color="blue",
        linestyle="--",
    )
    ax2_2.plot(
        data["epoch"],
        data["recon_weight"],
        label="recon_weight",
        color="red",
        linestyle="--",
    )
    ax2_2.plot(
        data["epoch"],
        data["smoothness_weight"],
        label="smoothness_weight",
        color="green",
        linestyle="--",
    )

    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax0_2.legend()

    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("distribution component value")

    ax[2].legend(loc="upper right")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("gradient magnitude")
    ax2_2.legend(loc="upper left")

    filename = f"{directory}/{data['start_time']}-training-history.png"
    plt.savefig(filename)

    plt.close()

    return filename


def gather(data, file):
    with open(file) as json_file:
        new_data = json.load(json_file)

    data["epoch"].append(new_data["epoch"])
    data["train_loss"].append(new_data["train_loss"])
    data["val_loss"].append(new_data["val_loss"])
    if new_data["saved"]:
        data["saved"].append(new_data["epoch"])

    data["alpha"].append(new_data["alpha"])
    data["beta"].append(new_data["beta"])
    data["weights"].append(new_data["weights"])

    data["lr"].append(new_data["lr"])

    data["bnll_loss"].append(new_data["bnll_loss"])
    data["recon_loss"].append(new_data["recon_loss"])
    data["smoothness_loss"].append(new_data["smoothness_loss"])

    data["bnll_loss_grad_magnitude"].append(new_data["bnll_loss_grad_magnitude"])
    data["recon_loss_grad_magnitude"].append(new_data["recon_loss_grad_magnitude"])
    data["smoothness_loss_grad_magnitude"].append(
        new_data["smoothness_loss_grad_magnitude"]
    )

    data["bnll_weight"].append(new_data["bnll_weight"])
    data["recon_weight"].append(new_data["recon_weight"])
    data["smoothness_weight"].append(new_data["smoothness_weight"])

    return data


def plot_training_history(files, directory="/tmp"):
    def initialize():
        return {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "saved": [],
            "alpha": [],
            "beta": [],
            "weights": [],
            "lr": [],
            "bnll_loss": [],
            "recon_loss": [],
            "smoothness_loss": [],
            "bnll_loss_grad_magnitude": [],
            "recon_loss_grad_magnitude": [],
            "smoothness_loss_grad_magnitude": [],
            "bnll_weight": [],
            "recon_weight": [],
            "smoothness_weight": [],
            "start_time": None,
        }

    data = initialize()

    plot_files = []
    current_start_time = None

    if len(files) == 0:
        print("No files to plot")
        return []

    for f in files:
        start_time = f.split("/")[-1].split("-")[0]
        if current_start_time is None:
            current_start_time = start_time
            data["start_time"] = start_time

        elif current_start_time != start_time:
            plot_files.append(plot(data, directory))
            data = initialize()
            current_start_time = start_time
            data["start_time"] = start_time
        data = gather(data, f)

    if len(data["epoch"]) > 0:
        plot_files.append(plot(data, directory))

    return plot_files
