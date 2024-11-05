from config import get_args
from plot import plot_training_history
from util import read_training_history

args = get_args(mode="plot-training")

if args.run_name is None:
    raise ValueError("Please provide a run name")

files = read_training_history(args.run_name, args.latest_only)
print("Read {} files".format(len(files)))
files = plot_training_history(files, directory=f"runs/{args.run_name}")

for file in files:
    print(f"Saved {file}")
