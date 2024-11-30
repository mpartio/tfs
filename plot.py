import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def plot_beta_predictions(alpha, beta, x, y, filename):
    alpha = alpha.squeeze()
    beta = beta.squeeze()
    x = x.squeeze()
    y = y.squeeze()

    plt.figure(figsize=(20, 10))

    plt.subplot(341)
    plt.imshow(x)
    plt.title("Input")
    plt.colorbar()

    plt.subplot(342)
    plt.imshow(y)
    plt.title("Truth")
    plt.colorbar()

    plt.subplot(343)
    mean = alpha / (alpha + beta)

    plt.imshow(mean)
    plt.title("Predicted Mean")
    plt.colorbar()

    plt.subplot(344)
    # Convert logvar to standard deviation
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = np.sqrt(var)

    plt.imshow(std)
    plt.title("Predicted Std")
    plt.colorbar()

    # Add a random sample from predictions
    # sample = torch.distributions.Beta(alpha, beta).sample((10,)).cpu()
    sample = np.random.beta(alpha, beta, size=((10,) + alpha.shape))
    plt.subplot(345)
    plt.imshow(sample[0])
    plt.title("One Random Sample")
    plt.colorbar()

    median = np.median(sample, axis=0)
    print(median.shape)
    plt.subplot(346)
    plt.imshow(median)
    plt.title("Median of Samples (n=10)")
    plt.colorbar()

    plt.subplot(347)
    plt.imshow(alpha)
    plt.title("Alpha")
    plt.colorbar()

    plt.subplot(348)
    plt.imshow(beta)
    plt.title("Beta")
    plt.colorbar()

    data = y - x
    cmap = plt.cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

    plt.subplot(349)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("True Diff")
    plt.colorbar()

    data = mean - x
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 4, 10)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Diff of Mean/L1={:.4f}".format(np.abs(mean, y).mean()))
    plt.colorbar()

    data = sample[0] - x
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 4, 11)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Diff of Sample/L1={:.4f}".format(np.abs(sample[0] - x).mean()))
    plt.colorbar()

    data = median - x
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 4, 12)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Diff of Median/L1={:.4f}".format(np.abs(median - x).mean()))
    plt.colorbar()

    plt.tight_layout()

    plt.savefig(filename)
    plt.close()
