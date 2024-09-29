import argparse

def parse_size(size):
    h, w = map(int, size.split('x'))
    return (h, w)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=12, metavar='N',
                        help='input batch size for testing (default: 12)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--loss-function', type=str, default='mae', metavar='LF',
                        help='loss function to use (default: mae)')
    parser.add_argument('--dataset-size', type=str, default="10k", metavar='N',
                        help='size of dataset to use (default: 10k)')
    parser.add_argument('--load-model-from', type=str, default='models/cc2-model.pth', metavar='file',
                        help='load model from file (default: models/cc2-model.pth)')
    parser.add_argument('--save-model-to', type=str, default='models/cc2-model.pth', metavar='file',
                        help='save model to file (default: models/cc2-model.pth)')
    parser.add_argument('--dim', type=int, default=192, metavar='N',
                        help='dimension of latent space (default: 192)')
    parser.add_argument('--patch-size', type=parse_size, default="4x4", metavar='NxN',
                        help='patch size hxw (default: 4x4)')
    return parser.parse_args()
