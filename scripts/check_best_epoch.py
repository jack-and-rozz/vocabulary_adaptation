# coding: utf-8
import argparse, random, sys
import glob
import subprocess as sp
from collections import defaultdict 
import pandas as pd

def check_log(name, path, num_unchanged_epochs=3):
    cmd1 = ["cat", path]
    cmd2 = ['grep', '-a', 'best_loss']
    cmd3 = ['cut', '-d', " ", '-f', '3,20,23']


    res1 = sp.Popen(cmd1, stdout=sp.PIPE)
    res2 = sp.Popen(cmd2, stdout=sp.PIPE, stdin=res1.stdout)
    res3 = sp.check_output(cmd3, stdin=res2.stdout)

    out = res3.decode('utf-8').strip().split('\n')

    if len(out) == 1:
        return (None, None) 
    #print(cmd1, cmd2, cmd3, file=sys.stderr)
    out = [[int(x) if i in [0,1] else float(x)
            for i, x in enumerate(l.split(' '))] for l in out]

    epochs, num_updates, best_losses = list(zip(*out))
    steps_per_epoch = num_updates[1] - num_updates[0]
    best_loss = min(best_losses)
    best_index = best_losses.index(best_loss)
    best_step = num_updates[best_index]

    # max_updates = num_updates[-1]
    max_updates = max(num_updates)
    delta = max_updates - best_step
    threshold = num_unchanged_epochs * steps_per_epoch
    suffix = '(!)' if delta <= threshold else ''
    if args.max_steps:
        max_updates = min(args.max_steps, max_updates)
        best_step = min(args.max_steps, best_step)
    # print(name, ': %d/%d' % (best_step, max_updates), 'best_loss=%.3f' % best_loss, suffix) 

    stepinfo = '%d/%d' % (best_step, max_updates)
    lossinfo = '%.3f' % best_loss
    return name, stepinfo, lossinfo, suffix
    # return best_step, nest_loss

def main(args):
    log_file_paths= sorted(glob.glob("%s/*/train.log" % (args.models_root)))

    stat = {}
    
    summary = defaultdict(dict)
    data = []
    for path in log_file_paths:
        name = path.split('/')[-2]
        d = check_log(name, path, args.num_unchanged_epochs)
        data.append(d)

    print()
    # if not args.max_steps:
    #     return

    header = ['Model', 'Steps', 'Loss', 'State']

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_rows", 101)
    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame(data, columns=header).set_index('Model')
    print()
    print(df)




if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-rt', '--models_root', default='checkpoints/latest', 
                        type=str)
    parser.add_argument('-ms', '--max_steps', default=None, type=int)
    parser.add_argument('-epoch', '--num_unchanged_epochs', default=5, type=int)
    args = parser.parse_args()
    main(args)

