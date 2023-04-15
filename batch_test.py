import os
import re
from numpy import percentile

REPEAT = 50
DIR_TO_SAVE = 'results/'
ALGOS = ['cpu-scan']
N_THREADS = [8]
N_ELEMS = [1000000]
EXEC = '"./build/Debug/prefix-sum.exe"'

def box(data):
    med, boxTop, boxBottom, whiskerTop, whiskerBottom = list(percentile(data, [50, 75, 25, 95, 5]))
    # IQR = boxTop - boxBottom
    # whiskerTop = boxTop + 1.5*IQR
    # whiskerBottom = boxBottom - 1.5*IQR
    return med, boxTop, boxBottom, whiskerTop, whiskerBottom

def save(algo_setting, n_elems, n_threads, data):
    filename = os.path.join(DIR_TO_SAVE, f'{algo_setting}-{n_elem}-{n_threads}.txt')
    with open(filename, 'a') as f:
        f.write(data)

for algo_setting in ALGOS:
    device, algo = algo_setting.split('-')
    for n_elem in N_ELEMS:
        for t in N_THREADS:
            command = EXEC + f' -d {device} -a {algo} -n {n_elem} -t {t} -r {REPEAT}'
            print(command)
            times_comp = []
            times_e2e = []
            with os.popen(command) as pipe:
                output = pipe.read()
            pattern = r'(\d+\.\d+) ms'
            result_list = {}
            try:
                result = [float(i) for i in re.findall(pattern, output)]
                # TODO: Parse CUDA result (has 6 time per iteration)
                assert len(result) == REPEAT * 2
            except:
                print('\nError when parsing, original output:\n', output)
                exit()
            times_comp = result[0::2]
            times_e2e = result[1::2]
            data = str(box(times_comp)) + "\n"
            save(algo_setting, n_elem, t, data)
            data = str(box(times_e2e)) + "\n"
            save(algo_setting, n_elem, t, data)
