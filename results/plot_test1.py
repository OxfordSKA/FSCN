# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from os.path import join
import os
import scipy
# import seaborn


def main():
    df = pd.read_csv('test1.txt', delim_whitespace=True, header=0)
    uniform = df.loc[df['weight'] == 'U']
    natural = df.loc[df['weight'] == 'N']

    # TODO(BM) plot the predicted noise..

    # seaborn.set_style('ticks')
    opts = dict(marker='.', ms=10)
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 8), nrows=2, sharex=False)
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.1)
    ax1.plot(uniform['num_times'], uniform['rms'], label='uniform', **opts)
    ax1.plot(natural['num_times'], natural['rms'], label='natural', **opts)
    ax1.legend(frameon=True)
    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0, uniform['num_times'].max()*1.1)
    ax1.set_ylabel('Image RMS (Jy/beam)')
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%i'))

    y = uniform['rms'].values / natural['rms'].values
    ax2.plot(uniform['num_times'], y, **opts)
    ax2.grid()
    ax2.set_xlabel('number of snapshots')
    ax2.set_ylabel('uniform rms / natural rms')
    ax2.set_xlabel('number of snapshots')
    ax2.set_xscale('log')
    ax2.set_xlim(0, uniform['num_times'].max() * 1.1)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%i'))
    fig.savefig('test1.eps')
    plt.close(fig)

if __name__ == '__main__':
    main()
