from os.path import join as ojoin
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re


def find_reg(pattern, rows):
    strings = re.findall(pattern, rows)
    arr = np.array(strings, dtype=float)
    return arr


def main(args):
    log_path = ojoin(args.log_path, "Training.log")
    model = args.log_path.split("/")[-1]

    # load the contents of the log file
    rows = open(log_path).read().strip()
    train_loss2 = find_reg(r"Loss2 (.*)   Acc1", rows)
    if len(train_loss2) > 0:
        train_loss = find_reg(r"Loss (.*)   Loss2", rows)
    else:
        train_loss = find_reg(r"Loss (.*)   Acc1", rows)
    train_acc1 = find_reg(r"Acc1 (.*)   Acc5", rows)
    steps = find_reg(r"Step: (.*)/", rows)
    steps = np.array(steps, dtype=int)
    total_step = np.max(steps)
    freq = np.min(steps) // 2

    epochs = find_reg(r"\[agedb_30\]\[(.*)\]Accuracy-Highest", rows)
    max_epoch = int(max(epochs))

    acc_lfw = find_reg(r"\[lfw\]\[[0-9]*\]Accuracy-Flip: (.*)\+-", rows) * 100
    max_acc_lfw = round(np.max(acc_lfw), 2)
    acc_agedb = find_reg(r"\[agedb_30\]\[[0-9]*\]Accuracy-Flip: (.*)\+-", rows) * 100
    max_acc_agedb = round(np.max(acc_agedb), 2)
    acc_cfpfp = find_reg(r"\[cfp_fp\]\[[0-9]*\]Accuracy-Flip: (.*)\+-", rows) * 100
    max_acc_cfpfp = round(np.max(acc_cfpfp), 2)
    acc_calfw = find_reg(r"\[calfw\]\[[0-9]*\]Accuracy-Flip: (.*)\+-", rows) * 100
    max_acc_caflw = round(np.max(acc_calfw), 2)
    acc_cplfw = find_reg(r"\[cplfw\]\[[0-9]*\]Accuracy-Flip: (.*)\+-", rows) * 100
    max_acc_cplfw = round(np.max(acc_cplfw), 2)

    loss_x = np.array(range(2 * freq, total_step + 1, freq))

    # plot the loss
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots()
    ax1.set_title(model)
    ax1.set_xlabel("Iteration #")

    ax1.plot(loss_x, train_loss, "r-", label="Loss")
    ax1.set_ylabel("Loss", color="r")
    if len(train_loss2) > 0:
        ax1.plot(loss_x, train_loss2, "g-", label="Loss2")

    ax2 = ax1.twinx()
    ax2.plot(loss_x, train_acc1, "b-", label="Acc 1")
    ax2.set_ylabel("Acc 1", color="b")

    fig.tight_layout()
    plt.savefig(ojoin(args.log_path, "Training.png"), format="png", dpi=600)
    plt.close()

    tick_freq = max_epoch // 20

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(acc_lfw, label=f"LFW: {max_acc_lfw}")
    plt.plot(acc_agedb, label=f"AgeDB-30: {max_acc_agedb}")
    plt.plot(acc_cfpfp, label=f"CFP-FP: {max_acc_cfpfp}")
    plt.plot(acc_calfw, label=f"CALFW: {max_acc_caflw}")
    plt.plot(acc_cplfw, label=f"CPLFW: {max_acc_cplfw}")
    plt.title(model)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.ylim(40, 95)
    plt.xticks(
        np.arange(1, max_epoch + 1, tick_freq), np.arange(2, max_epoch + 2, tick_freq)
    )
    plt.legend(loc="lower right")
    plt.savefig(ojoin(args.log_path, "Validation.png"), format="png", dpi=600)
    plt.close()

    print("Successfully plotted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Plot Training Logs")
    parser.add_argument(
        "--log_path",
        type=str,
        default="output/DA_Back/DABack_1024D_Head",
        help="folder path to log file",
    )
    args = parser.parse_args()
    main(args)
