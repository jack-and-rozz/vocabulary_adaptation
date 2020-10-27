# -*- coding:utf-8 -*-
#from __future__ import print_function
import sys, os, time, re, math, collections, json
import random, string
import numpy as np
import itertools, datetime
from itertools import chain
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
import multiprocessing as mp


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BLACK = "\033[30m"
UNDERLINE = '\033[4m'
BOLD = BLACK+"\033[1m" #UNDERLINE
RESET = "\033[0m"

############################################
#       Vector
############################################

from inspect import currentframe
def dbgprint(*args):
  names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
  print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

############################################
#       Vector
############################################

def normalize_vector(v):
  norm = np.linalg.norm(v)
  if norm > 0:
    return v / norm
  else:
    return v

############################################
#        Dictionary
############################################

# dictionaryのkey:value入れ替え
def invert_dict(dictionary):
    return {v:k for k, v in list(dictionary.items())}

# 辞書のvalueでソート。デフォルトは降順
def sort_dict(dic, sort_type="DESC"):
    counter = collections.Counter(dic)
    if sort_type == "ASC":
        count_pairs = sorted(list(counter.items()), key=lambda x: x[1])
    elif sort_type == "DESC":
        count_pairs = sorted(list(counter.items()), key=lambda x: -x[1])
    key, value = list(zip(*count_pairs))
    return (key, value)

# funcは key, valueのタプルを引数に取り、同じく key, valueのタプルを返す関数
def map_dict(func, _dict):
    return dict([func(k,v) for k,v in list(_dict.items())]) 


def split_dict():
    pass

class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, key):
    if key in self:
      return self[key]
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

class recDotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def __init__(self, _dict={}):
    for k in _dict:
      if isinstance(_dict[k], dict):
        _dict[k] = recDotDict(_dict[k])
      if isinstance(_dict[k], list):
        for i,x in enumerate(_dict[k]):
          if isinstance(x, dict):
            _dict[k][i] = dotDict(x)
    super(recDotDict, self).__init__(_dict)

  def __getattr__(self, key):
    if key in self:
      return self[key]
    # else:
    #   return None
    raise AttributeError("\'%s\' is not in %s" % (str(key), str(self.keys())))

class rec_defaultdict(collections.defaultdict):
  def __init__(self):
    self.default_factory = type(self)

class recDotDefaultDict(collections.defaultdict):
  __getattr__ = collections.defaultdict.__getitem__
  __setattr__ = collections.defaultdict.__setitem__
  __delattr__ = collections.defaultdict.__delitem__
  def __init__(self, _=None):
    super(recDotDefaultDict, self).__init__(recDotDefaultDict)

def flatten_recdict(d):
  res = dotDict()
  for k in d:
    if isinstance(d[k], dict):
      subtrees = flatten_recdict(d[k])
      for kk, v in subtrees.items():
        res['%s.%s' % (k, kk)] = v
    else:
      res[k] = d[k]
  return res

def flatten_batch(batch):
  '''
  Decompose a batch, a tree where each value-list containing all those of all examples is tied with their keys into a list of trees containing pairs of key-values, separated by examples. 

  e.g.
  {'a': [1, 10], 'b': [2, 20]} -> [{'a': 1, 'b':2}, {'a':10, 'b':20}]
  {'a': [1, 10], 'b': [2, 20, 30]} -> [{'a': 1, 'b':2}, {'a':10, 'b':20}, {'b': 30}]

  <Args>
  - batch: a tree (a hierarchical dictionary).
  <Return>
  - entries: a list of trees.

  '''
  entries = []
  for key in batch:
    if isinstance(batch[key], dict): 
      # If subtree is a dict, apply flatten.
      subtrees = flatten_batch(batch[key])
    elif hasattr(batch[key], '__iter__'): 
      # If subtree is iterable, convert it as a list.
      subtrees = [value for value in batch[key]]
    else:  
      # If not iterable, ignored.
      subtrees = None

    if not subtrees:
      continue

    if len(entries) < len(subtrees):
      entries += [recDotDefaultDict() for _ in range(len(subtrees) - len(entries))]

    for i, sbt in enumerate(subtrees):
      entries[i][key] = sbt

  return entries # A list of trees.

def batching_dicts(batch, d):
  '''
  Args:
  - batch: 
  Recursively add to batch an entry whose type is recDotDict.
  e.g. [{'a': 1, 'b':2}, {'a':10, 'b':20}] -> {'a': [1, 10], 'b': [2, 20]}
  '''
  try:
    assert type(d) in [recDotDefaultDict, recDotDict]
    assert type(batch) in [recDotDefaultDict, recDotDict]
  except:
    sys.stderr.write('The two arguments must be instances of recDotDefaultDict, but batch, data = (%s, %s).\n' % (type(batch), type(d)))
    sys.stderr.write(str(d.keys()) + '\n')
    exit(1)
  for k in d:
    if isinstance(d[k], dict):
      batching_dicts(batch[k], d[k])
    else:
      if batch[k]:
        batch[k].append(d[k])
      else:
        batch[k] = [d[k]]
  return batch # A tree of lists.

def modulize(dictionary):
    import imp
    m = imp.new_module("")
    m.__dict__.update(dictionary)
    return m


############################################
#        Batching
############################################

def zero_one_vector(one_indices, n_elem):
  l = [0.0 for _ in range(n_elem)]
  for idx in one_indices:
    l[idx] = 1.0
  return l

def batching(data, batch_size):
  batch = [[x[1] for x in d2] for j, d2 in itertools.groupby(enumerate(data), lambda x: x[0] // batch_size)]
  return batch

############################################
#        Random
############################################

# arr = [arr0, arr1, ...]
# size_arr : [len(arr0), len(arr1), ...]
def random_select_by_scale(size_arr):
  scale = [sum(size_arr[:i + 1]) / float(sum(size_arr)) for i in range(len(size_arr))]
  random_number_01 = np.random.random_sample()
  _id = min([i for i in range(len(scale))
             if scale[i] > random_number_01])
  return _id

############################################
#        Text Formatting
############################################

def format_zen_han(l):
  import mojimoji
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  return l

def format_zen_han(l):
  import mojimoji
  l = l.decode('utf-8') if type(l) == str else l
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  l = l.encode('utf-8')
  return l


############################################
#        Logging
############################################

def logManager(logger_name='main', 
               handler=StreamHandler(),
               log_format = "[%(levelname)s] %(asctime)s - %(message)s",
               #log_format = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)',
               level=DEBUG):
    formatter = Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(logger_name)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


############################################
#        File reading
############################################

def read_file(file_path, type_f=str, do_flatten=False, replace_patterns=None,
              do_tokenize=True, delim=' ', max_rows=None):
  lines = []
  for i, line in enumerate(open(file_path, "r")):
    if max_rows and i > max_rows:
      break
    line = line.replace("\n", "")
    if replace_patterns:
      for (before, after) in replace_patterns:
        line = line.replace(before,  after)
    lines.append(line)

  if do_tokenize:
    lines = [[type_f(t) for t in line.split(delim) if t != ''] for line in lines]

  if do_flatten:
    lines = flatten(lines)
  return lines

def read_file_with_header(file_path, type_f=str, do_flatten=False, delim=' ', do_tokenize=True):
    data = read_file(file_path, type_f, do_flatten, delim, do_tokenize)
    header = data[0]
    data = data[1:]
    return header, data

def read_vector(file_path, skip_first, type_f=float):
    # あんまりサイズが大きくなるとエラー？
    lines = read_file(file_path)
    vector_dict = collections.OrderedDict({})
    # 初めの一行目はベクトルの行数と次元なのでスキップ
    start = 1 if skip_first else 0
    for l in lines[start:]:
        vector_dict[l[0]] = np.array([type_f(w) for w in l[1:]])
    return vector_dict


############################################
#        String
############################################

# argparserの文字列をboolに変換
def str2bool(str_):
  bool_ = True
  str_ = str_.lower()
  if str_ in ["t", "true", "1", True]:
    return True
  elif str_ in ["f", "false", "0", False]:
    return False
  else:
    print("Irregular bool string")
    exit(1)


def separate_path_and_filename(file_path):
    pattern = '^(.+)/(.+)$'
    m = re.match(pattern, file_path)
    if m:
      path, filename = m.group(1), m.group(2) 
    else:
      path, filename = None , file_path
    return path, filename

def count_uniq_tokens(sentence_array):
    vocab = collections.defaultdict(int)
    for s in sentence_array:
        for token in s:
            vocab[token] += 1
    return vocab

# 数値文字列の行列 ([['1', '3'], ['2', '4']]) を数値に 
# read_file 時に指定すればいいのでは？
#def strmat_to_type(mat, type_f):
#    res = [[type_f(i) for i in l] for l in mat]
#    return res

def get_random_string(length):
  return ''.join(random.choices(string.ascii_letters, k=length))

############################################
#        List
############################################

def regexp_indices(pattern, l):
    """
    リストの中で正規表現にマッチしたものをすべて返す
    pattern: regexp
    l      : list
    """
    res = []
    for i, x in enumerate(l):
        m = re.match(pattern, x)
        if not m == None:
            res.append(i)
    return res

def devide_by_labels(elements, labels, N):
    """
    elementsを付与されたラベルlabelsで分けた結果を返す
    elements: list
    labels  : list
    N       : int
    """
    res = [[] for _ in range(0, N)]
    for i, label in enumerate(labels):
        res[label].append(elements[i])
    return res

def find_first(predicate, l):
    """
    条件を満たす最初の要素を返す
    predicate: function returns bool
    l: list
    """
    if len(l) == 0:
        return 
    if predicate(l[0]):
        return l[0]
    else:
        find_first(predicate, l[1:])


def flatten(l):
  return list(chain.from_iterable(l))


def flatten_with_idx(tensor):
  res = flatten([[(i, x) for x in t] for i, t in enumerate(tensor)])
  return list(map(list, list(zip(*res)))) # in-batch indices, tensor

def max_elem( lis ):
  L = lis[:]#copy
  S = set(lis)
  S = list(S)
  MaxCount=0
  ret='nothing...'

  for elem in S:
    c=0
    while elem in L:
      ind = L.index(elem)
      foo = L.pop(ind)
      c+=1
    if c>MaxCount:
      MaxCount=c
      ret = elem
  return ret

############################################
#        Measuring
############################################

def benchmark(func=None, prec=3, unit='auto', name_width=0, time_width=8):
    """
    A decorator that prints the time a function takes
    to execute per call and cumulative total.

    Accepts the following keyword arguments:

    `unit`  str  time unit for display. one of `[auto, us, ms, s, m]`.
    `prec`  int  radix point precision.
    `name_width`  int  width of the right-aligned function name field.
    `time_width`  int  width of the right-aligned time value field.

    For convenience you can also set attributes on the benchmark
    function itself with the same name as the keyword arguments
    and the value of those will be used instead. This saves you
    from having to call the decorator with the same arguments each
    time you use it. Just set, for example, `benchmark.prec = 5`
    before you use the decorator for the first time.
    """
    import time
    if hasattr(benchmark, 'prec'):
        prec = getattr(benchmark, 'prec')
    if hasattr(benchmark, 'unit'):
        unit = getattr(benchmark, 'unit')
    if hasattr(benchmark, 'name_width'):
        name_width = getattr(benchmark, 'name_width')
    if hasattr(benchmark, 'time_width'):
        time_width = getattr(benchmark, 'time_width')
    if func is None:
        return partial(benchmark, prec=prec, unit=unit,
                       name_width=name_width, time_width=time_width)

    @wraps(func)
    def wrapper(*args, **kwargs):  # IGNORE:W0613
        def _get_unit_mult(val, unit):
            multipliers = {'us': 1000000.0, 'ms': 1000.0, 's': 1.0, 'm': (1.0 / 60.0)}
            if unit in multipliers:
                mult = multipliers[unit]
            else:  # auto
                if val >= 60.0:
                    unit = "m"
                elif val >= 1.0:
                    unit = "s"
                elif val <= 0.001:
                    unit = "us"
                else:
                    unit = "ms"
                mult = multipliers[unit]
            return (unit, mult)
        t = time.clock()
        res = func(*args, **kwargs)
        td = (time.clock() - t)
        wrapper.total += td
        wrapper.count += 1
        tt = wrapper.total
        cn = wrapper.count
        tdu, tdm = _get_unit_mult(td, unit)
        ttu, ttm = _get_unit_mult(tt, unit)
        td *= tdm
        tt *= ttm
        print((" -> {0:>{8}}() @ {1:>03}: {3:>{7}.{2}f} {4:>2}, total: {5:>{7}.{2}f} {6:>2}"
              .format(func.__name__, cn, prec, td, tdu, tt, ttu, time_width, name_width)))
        return res
    wrapper.total = 0
    wrapper.count = 0
    return wrapper

# 時間を測るデコレータ
def timewatch(logger=None):
  if logger is None:
    logger = logManager(logger_name='utils')
  def _timewatch(func):
    def wrapper(*args, **kwargs):
      start = time.time()
      result = func(*args, **kwargs)
      end = time.time()
      logger.info("%s: %f sec" % (func.__name__ , end - start))
      return result
    return wrapper
  return _timewatch


############################################
#        MultiProcessing
############################################

def multi_process(func, *args):
  def wrapper(_func, idx, q):
    def _wrapper(*args, **kwargs):
      res = func(*args, **kwargs)
      return q.put((idx, res))
    return _wrapper

  workers = []
  # mp.Queue() seems to have a bug..? 
  # (stackoverflow.com/questions/13649625/multiprocessing-in-python-blocked)
  q = mp.Manager().Queue() 
  
  # kwargs are not supported... (todo)
  for i, a in enumerate(zip(*args)):
    worker = mp.Process(target=wrapper(func, i, q), args=a)
    workers.append(worker)
    worker.daemon = True  # make interrupting the process with ctrl+c easier
    worker.start()

  for worker in workers:
    worker.join()
  results = []
  while not q.empty():
    res = q.get()
    results.append(res)
  
  return [res for i, res in sorted(results, key=lambda x: x[0])]

def argwrapper(args):
    '''
    args.append((func, arg1, arg2, ...))
    p = multiprocessing.Pool(n_process)
    res = p.map(argwrapper, args)

    '''
    return args[0](*args[1:])


#############################################
#        Others
#############################################

def decorate_instance_methods(func_list, decorator):
  for func in func_list:
    func = decorater(func)


def colored(str_, color):
  '''
  Args: colors: a str or list of it.
  '''
  ctable = {
    'RESET' : "\033[0m",
    'black': "\033[30m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'purple': "\033[35m",
    'underline': '\033[4m',
    'link': "\033[31m" + '\033[4m',
    'bold': '\033[30m' + "\033[1m",
  }

  if type(color) == str:
    res = ctable[color] + str_ + ctable['RESET']
  elif type(color) == tuple or type(color) == list:
    res = "".join([ctable[c] for c in color]) + str_ + CTABLE['RESET']
  return res 


def timestamp():
  #return datetime.datetime.strptime('2017-11-23 15:00', "%Y-%m-%d %H:%M").strftime('%s')
  return datetime.datetime.now().strftime("%Y-%m%d-%H%M%S")


############################################
#           Json
############################################

def read_text(source_path, max_rows=0):
  data = []
  for i, l in enumerate(open(source_path)):
    if max_rows and i >= max_rows:
      break
    data.append(l)
  return data

def read_jsonlines(source_path, max_rows=0):
  data = []#collections.OrderedDict()
  for i, l in enumerate(open(source_path)):
    if max_rows and i >= max_rows:
      break
    d = recDotDict(json.loads(l))
    data.append(d)
  return data

def read_json(source_path):
  data = json.load(open(source_path)) 
  return recDotDict(data)

def dump_as_json(entities, file_path, as_jsonlines=True):
  if as_jsonlines:
    if os.path.exists(file_path):
      os.system('rm %s' % file_path)
    with open(file_path, 'a') as f:
      for entity in entities.values():
        json.dump(entity, f, ensure_ascii=False)
        f.write('\n')
  else:
    with open(file_path, 'w') as f:
      json.dump(entities, f, indent=4, ensure_ascii=False)


############################################
#          Debug
############################################

def print_batch(batch):
  #Debug of batch
  for k, v in flatten_recdict(batch).items():
    if isinstance(v, np.ndarray):
      print(k, v.shape)
    else:
      print(k)

