# coding: utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
#import seaborn  as plt # sns

# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'Calibri'

# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = '\usepackage{sfmath}'

def main(args):
    df = pd.read_csv(args.log_file)
    print(df)
    src = args.src_domain
    tgt = args.tgt_domain
    direction = "%s2%s" % (src, tgt)
    # data_sizes = list(df.columns)[1:]
    data_sizes = [10000, 100000, 1000000, 2000000]
    # print(data_sizes)
    target_models= [
        (tgt + '.baseline', 'In-domain'),
        (direction + '.finetune.v_' + src, 'Fine-tuning'),
        (direction + '.multidomain.domainmixing', 'Domain token mixing'),
        #(direction + '.multidomain.domainweighting', 'Domain weighting'),
        (direction + '.finetune.idt', 'Proposed (Linear)'),
        (direction + '.finetune.llm', 'Proposed (LLM)'),
    ]
    model_ids, model_names = list(zip(*target_models))
    #print(df.query('Model == %s' % model_ids[1]))
    #print(df.loc[model_ids[0]])
    x = data_sizes
    # for model_id in model_ids:
    #     y = df[df['Model'] == model_id].tolist()
    #     print(y[1:])
    #     exit(1)
    data = OrderedDict()
    for row in df.values.tolist():
        model_id = row[0]
        bleus = row[1:]
        if model_id not in model_ids:
            continue
        data[model_id] = bleus


    label_size= 16
    legend_size=12.5
    params = {
        'axes.labelsize': label_size, # なにが変わるのかよくわからない
        # 'text.fontsize': 16, # テキストサイズだろう
        'legend.fontsize': legend_size, # 凡例の文字の大きさ
        'xtick.labelsize': label_size, # x軸の数値の文字の大きさ
        'ytick.labelsize': label_size, # y軸の数値の文字の大きさ
        #'text.usetex': True, # 使用するフォントをtex用（Type1）に変更
        #'figure.figsize': [width, height]
    } 
    plt.rcParams.update(params)

    marker_type=['o', 'o', 'o', '^', 'o', 's']
    linestyle=[':', '-.', '--', '--', '-', '-']
    # marker_type=['x', '*', '^', 'o', 's']
    # linestyle=[':', '-.', '--', '-', '-']
    for i, (model_id, model_name) in enumerate(target_models):
        y = data[model_id]
        plt.plot(x, y, marker=marker_type[i], linestyle=linestyle[i]) 
    # ax = plt.gca()
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    plt.legend(model_names, loc='upper left')
    plt.title("")
    # plt.xticks([10**4, 10**5, 10**6, 2*10**6], ['10k', '100k{\\small (Table 3)}', '1000k', '2000k'])
    plt.xscale('log')
    plt.xticks([10**4, 10**5, 10**6, 2*10**6], ['10k', '100k{\\normalsize (Table 3)}', '1000k', '2000k'])
    plt.xlabel("Size of target-domain parallel data")
    plt.ylabel("BLEU score")
    # plt.grid()
    # plt.show()
    #for model_id, model_name in target_models:
    output_file = args.output_file if args.output_file else '%s2%s-by-datasize.pdf' % (src, tgt)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    pass


if __name__ == "__main__":
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('log_file', type=str)
    parser.add_argument('-src', '--src_domain', default='jesc', type=str)
    parser.add_argument('-tgt', '--tgt_domain', default='aspec', type=str)
    parser.add_argument('-out', '--output_file', default=None, type=str)
    args = parser.parse_args()
    main(args)


"""
aspec,,,,37.73
aspec.baseline,3.59,10.09,27.04,37.73
aspec_sp.baseline,,11.06,,37.02
jesc.baseline,,,,4.36
jesc2aspec.finetune.idt,11.83,18.18,30.14,37.32
jesc2aspec.finetune.llm-new,11.51,23.96,38.87,40.40
jesc2aspec.finetune.llm,10.64,23.65,38.23,40.19
jesc2aspec.finetune.v_jesc,1.36,5.30,21.31,26.37
jesc2aspec.multidomain.domainmixing,11.24,17.98,31.06,36.54
jesc2aspec.multidomain.domainweighting,10.84,17.98,31.71,37.13
jesc_sp.baseline,,,,3.50
"""
