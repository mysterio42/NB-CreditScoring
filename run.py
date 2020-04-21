import argparse

from utils.data import load_data
from utils.model import train_model, training_methods, load_model
from utils.plot import plot_data


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  '
                             'False: Train model default: False')

    parser.add_argument('--method', choices=list(training_methods.keys()),
                        help='Training methods: cv-  Cross-Validation  default 10-Fold'
                             'split- default 70 perc train, 30 perc test')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.load:
        model = load_model()
        income, age, loan = 27845.8008938469, 55.4968525394797, 10871.1867897838
        print(model.predict([list((income, age, loan))])[0])

    else:
        features, labels = load_data('data/credit_data.csv', 'default', ('income', 'age', 'loan'))
        plot_data(features, labels)
        model = train_model(features, labels, args.method)
        income, age, loan = 27845.8008938469, 55.4968525394797, 10871.1867897838
        print(model.predict([list((income, age, loan))])[0])
