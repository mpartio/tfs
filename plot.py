import json
import matplotlib.pyplot as plt
import numpy as np


def plot(data, directory):
    # print("Plotting", start_time, "length", len(data["epoch"]))

    loss_components = int("loss_components" in data)
    gradient_components = int("gradients" in data)

    fig, ax = plt.subplots(
        1, 1 + loss_components + gradient_components, figsize=(18, 6)
    )

    ax[0].plot(data["epoch"], data["train_loss"], label="train_loss")
    ax[0].plot(data["epoch"], data["val_loss"], label="val_loss")

    if loss_components:
        for k, v in data["loss_components"].items():
            if "total" in k:
                name = "_".join(k.split("_")[1:])
                ax[0].plot(data["epoch"], v, label=name, linestyle="--")

    ax[0].scatter(
        data["saved"],
        [data["val_loss"][x - 1] for x in data["saved"]],
        label="saved",
        color="red",
    )

    ax0_2 = ax[0].twinx()

    lr = [x * 1e6 for x in data["lr"]]
    ax0_2.plot(data["epoch"], lr, label="lr * 1e6", color="green")

    num_mix = len(data["distribution_components"]["alpha"][0])

    colors = ["red", "blue", "green", "brown"]
    linestyles = ["-", "--", ":", "-."]
    for i in range(num_mix):
        for j, k in enumerate(data["distribution_components"].keys()):
            d = np.asarray(data["distribution_components"][k])
            ax[1].plot(
                data["epoch"],
                d[:, i],
                label=f"{k}_{i}",
                color=colors[i],
                linestyle=linestyles[j],
            )

    for k, v in data["gradients"].items():
        ax[2].plot(data["epoch"], v, label=f"{k} gradients")

    ax2_2 = ax[2].twinx()

    for k, v in data["loss_components"].items():
        if "weight_" in k:
            ax2_2.plot(data["epoch"], v, label=k, linestyle="--")

    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax0_2.legend(loc="lower left")
    ax0_2.set_ylabel("learning rate * 1e6")

    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("distribution component value")

    ax[2].legend(loc="upper right")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("gradient magnitude")
    ax2_2.legend(loc="upper left")
    ax2_2.set_ylabel("loss weight")

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
    data["lr"].append(new_data["lr"])

    if new_data["saved"]:
        data["saved"].append(new_data["epoch"])

    for k, v in new_data["distribution_components"].items():
        try:
            data["distribution_components"][k].append(v)
        except KeyError:
            data["distribution_components"][k] = [v]

    if "loss_components" in new_data:
        for k, v in new_data["loss_components"].items():
            try:
                data["loss_components"][k].append(new_data["loss_components"][k])
            except KeyError:
                data["loss_components"][k] = [new_data["loss_components"][k]]

    if "gradients" in new_data:
        for k, v in new_data["gradients"].items():
            try:
                data["gradients"][k].append(new_data["gradients"][k])
            except KeyError:
                data["gradients"][k] = [new_data["gradients"][k]]

    return data


def plot_training_history(files, directory="/tmp"):
    def initialize():
        return {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "saved": [],
            "distribution_components": {},
            "lr": [],
            "loss_components": {},
            "gradients": {},
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
