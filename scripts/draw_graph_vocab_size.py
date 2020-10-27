import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
sns.set_style('white')

title_size = 40
label_size= 35
legend_size = 30
tick_size = 30
font_size = 28
marker_size = 20
line_width = 6
ymin = 20
ymax = 31

params = {
    'axes.titlesize': title_size,
    'axes.labelsize': label_size,
    'legend.fontsize': legend_size,
    'xtick.labelsize': tick_size,
    'ytick.labelsize': tick_size,
    'lines.markersize': marker_size,
    'lines.linewidth': line_width,
    'font.size': font_size,
    'legend.facecolor': 'white',
    'figure.subplot.wspace': 0.05
}

plt.rcParams.update(params)

def plot(ax, data, baseline, shift_xs, shift_ys, title, model_names, legend='none', xticks=True):
    # ax.xticks([10**4, 10**5, 10**6, 2*10**6], ['10k', '100k{\\normalsize (Table 3)}', '1000k', '2000k'])
    marker_types = ['o', '^']
    line_types = ['-', '--']
    vocab_sizes = [2000, 4000, 8000, 16000, 32000]
    xlabels = ['2k', '4k', '8k', '16k', '32k']
    x = vocab_sizes

    for y, marker_type, line_type, shift_x, shift_y in zip(data, marker_types, line_types, shift_xs, shift_ys):
        ax.plot(x, y, line_type, marker=marker_type)

        for x_i, y_i, shift_x_i, shift_y_i in zip(x, y, shift_x, shift_y):
            ax.text(x_i+shift_x_i, y_i+shift_y_i, f'{y_i:.2f}')

    # Log scaleなので、掛け算で最小と最大を計算
    xmin = np.min(vocab_sizes) * (1/1.2)
    xmax = np.max(vocab_sizes) * 1.2

    ax.plot([xmin, xmax], [baseline, baseline], ':', c='k')
    ax.text(2000, baseline-1, 'FT-srcV (baseline)', fontsize=30)

    # Scaleをlogに
    ax.set_xscale('log')

    # Tickの設定
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels(xlabels)

    # 表示区域の設定
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if legend != 'none':
        ax.legend(model_names, loc=legend)

    ax.set_title(title)

def main(args):
    # enja1 = [22.81, 24.78, 24.46, 25.30, 25.17] # w/ raw corpus
    # enja2 = [22.84, 23.72, 22.74, 22.86, 21.36] # w/o raw corpus
    # deen1 = [26.22, 27.78, 27.66, 28.04, 27.61]
    # deen2 = [26.7, 27.24, 27.03, 26.81, 24.74]

    # # Baselines
    # enja_ft_srcV = 21.70
    # deen_ft_srcV = 24.84


    enja1 = [23.57, 23.43, 23.12, 21.79, 20.95] # w/o raw corpus
    enja2 = [23.42, 23.65, 23.81, 25.31, 24.91] # w/ raw corpus
    deen1 = [26.24, 27.59, 26.87, 26.4, 25.7]
    deen2 = [26.79, 27.77, 27.55, 27.87, 27.04]
    # Baselines
    enja_ft_srcV = 21.45
    deen_ft_srcV = 24.59

    # テキストの位置調整
    enja1_shift_x = [-200, -1000, -2500, -5500, -10500]
    enja1_shift_y = [0.6, -1.2] + [-0.9, -1.0, -0.9]
    enja2_shift_x = [-200, -1000, -2200, -5000, -10500]
    enja2_shift_y = [-0.9] + [0.5 for i in range(4)]
    enja_shift_x = [enja1_shift_x, enja2_shift_x]
    enja_shift_y = [enja1_shift_y, enja2_shift_y]

    deen1_shift_x = [-200, -1000, -2000, -5000, -10500]
    deen1_shift_y = [1.2] + [-1.1, -1.1, -1.1, -0.9]
    deen2_shift_x = [-200, -1000, -2000, -5000, -10500]
    deen2_shift_y = [-1.4] + [0.5, 0.5, 0.5, 0.5]
    deen_shift_x = [deen1_shift_x, deen2_shift_x]
    deen_shift_y = [deen1_shift_y, deen2_shift_y]
    # enja1_shift_x = [-200, -1300, -2000, -5000, -10500]
    # enja1_shift_y = [-0.9] + [0.5 for i in range(4)]
    # enja2_shift_x = [-200, -1000, -2500, -5500, -10500]
    # enja2_shift_y = [0.6, -1.2] + [-0.9 for i in range(3)]
    # enja_shift_x = [enja1_shift_x, enja2_shift_x]
    # enja_shift_y = [enja1_shift_y, enja2_shift_y]

    # deen1_shift_x = [-200, -1300, -2000, -5000, -10500]
    # deen1_shift_y = [-0.9] + [0.5 for i in range(4)]
    # deen2_shift_x = [-200, -1300, -2000, -5000, -10500]
    # deen2_shift_y = [.6] + [-1.1 for i in range(4)]
    # deen_shift_x = [deen1_shift_x, deen2_shift_x]
    # deen_shift_y = [deen1_shift_y, deen2_shift_y]

    
    
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(20, 8), squeeze=True)

        # model_names =  ['VA-LLM', 'VA-LLM \n w/o monolingual data']
        model_names =  ['VA-LLM (w/o monolingual)', 'VA-LLM (w/ monolingual)']
        plot(axes[0], [enja1, enja2], enja_ft_srcV, enja_shift_x, enja_shift_y, 'En-Ja', model_names, legend='upper left')
        plot(axes[1], [deen1, deen2], deen_ft_srcV, deen_shift_x, deen_shift_y, 'De-En', model_names)
    
    # 共通のlabel用
    ax = fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Target-domain vocabulary size")
    plt.ylabel("BLEU score")

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    #ax.set_xlabel("Target-domain vocabulary size")
    #ax.set_ylabel("BLEU score")
    plt.savefig('vocab_size.pdf', bbox_inches="tight", pad_inches=0.03)
    # plt.show()

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    args = parser.parse_args()
    main(args)
