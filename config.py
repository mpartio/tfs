import argparse

def parse_size(size):
    h, w = map(int, size.split('x'))
    return (h, w)

def get_args(mode = None):
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--dataset-size', type=str, default="10k", metavar='N',
                        help='size of dataset to use (default: 10k)')
    parser.add_argument('--dim', type=int, default=192, metavar='N',
                        help='dimension of latent space (default: 192)')
    parser.add_argument('--patch-size', type=parse_size, default="4x4", metavar='NxN',
                        help='patch size hxw (default: 4x4)')
    parser.add_argument('--run-name', type=str, default=None, metavar='name',
                        help='name of run (default: None)')
    parser.add_argument('--num-mixtures', type=int, default=1, metavar='N',
                        help='number of mixtures (default: 1)')

    if mode == "plot-training":
        parser.add_argument("--latest-only", action="store_true", help="plot latest run only")

    return parser.parse_args()
