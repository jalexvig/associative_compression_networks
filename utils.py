import re
import matplotlib.pyplot as plt


def get_losses_from_log(fpath):

    with open(fpath) as f:
        train_losses = re.findall('(\d+) Average loss: (.*)', f.read())

    res = [(idx, float(val)) for idx, val in train_losses]

    return res


def plot_loss(fpath1: str,
              label1: str,
              fpath2: str='',
              label2: str=''):


    losses1 = get_losses_from_log(fpath1)
    _, vals1 = zip(*losses1)
    plt.plot(vals1, label=label1)

    if fpath2:
        losses2 = get_losses_from_log(fpath2)
        _, vals2 = zip(*losses2)
        plt.plot(vals2, label=label2)

    plt.xlabel('Epoch')
    plt.ylabel('Nats')

    title = 'Training Loss for {}'.format(label1) + ' vs. {}'.format(label2) if label2 else ''
    plt.title(title)

    plt.legend()

    plt.show()


if __name__ == '__main__':

    fpath1 = './models/archive/512_hidden_dim/log_vae.txt'
    fpath2 = './models/archive/512_hidden_dim/log.txt'

    plot_loss(fpath1, 'VAE', fpath2, 'ACN')
