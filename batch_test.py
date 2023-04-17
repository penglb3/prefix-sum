import os
import re
from numpy import percentile

REPEAT = 50
DIR_TO_SAVE = 'results/'
ALGOS = ['cpu-seq','cpu-scan','cpu-efficient','cpu-block','cuda-scan','cuda-efficient']
X_OFFSET = [i * 0.1 for i in range(len(ALGOS))]
N_THREADS = [8] # algos sensitive to this: cpu-block and cpu-efficient
X_BASE = [1, 2, 3, 4]
N_ELEMS = [1024 * 2 ** (5 * i) for i in X_BASE]
EXEC = '"./build/Debug/prefix-sum.exe"'

def box(data):
    med, boxTop, boxBottom, whiskerTop, whiskerBottom = list(percentile(data, [50, 75, 25, 95, 5]))
    # IQR = boxTop - boxBottom
    # whiskerTop = boxTop + 1.5*IQR
    # whiskerBottom = boxBottom - 1.5*IQR
    return med, boxTop, boxBottom, whiskerTop, whiskerBottom

def formated_box_plot(data):
    return " ".join([f'{i:.3f}' for i in box(data)]) + "\n"

def save_name(algo_setting, n_elem, n_threads):
    return os.path.join(DIR_TO_SAVE, f'{algo_setting}-{n_elem}-{n_threads}.txt')

def save(algo_setting, n_elem, n_threads, data):
    filename = save_name(algo_setting, n_elem, n_threads)
    with open(filename, 'a') as f:
        f.write(data)

def run(device, algo, n_elem, t):
    command = EXEC + f' -d {device} -a {algo} -n {n_elem} -t {t} -r {REPEAT}'
    print(command)
    times_comp = []
    times_e2e = []
    with os.popen(command) as pipe:
        output = pipe.read()
    comp_pattern = r'\[C\] (\d+\.\d+) ms'
    e2e_pattern = r'End to end latency: (\d+\.\d+) ms'
    try:
        times_comp = [float(i) for i in re.findall(comp_pattern, output)][1:]
        times_e2e = [float(i) for i in re.findall(e2e_pattern, output)][1:]
        assert len(times_comp) == REPEAT and len(times_e2e) == REPEAT
    except:
        print('\nError when parsing, original output:\n', output)
        exit()
    data_comp = formated_box_plot(times_comp)
    save(algo_setting, n_elem, t, data_comp)
    data_e2e = formated_box_plot(times_e2e)
    save(algo_setting, n_elem, t, data_e2e)
    return data_comp, data_e2e


for i_offset, algo_setting in enumerate(ALGOS):
    comps = []
    e2es = []
    device, algo = algo_setting.split('-')
    for i_base, n_elem in enumerate(N_ELEMS):
        for t in N_THREADS:
            result_fname = save_name(algo_setting, n_elem, t)
            if(not os.path.isfile(result_fname)):
                data_comp, data_e2e = run(device, algo, n_elem, t)
            else:
                with open(result_fname, 'r') as f:
                    data_comp, data_e2e = f.readlines()
            x = X_BASE[i_base] + X_OFFSET[i_offset]
            comps.append(f"{x:.2f} " + data_comp)
            e2es.append(f"{x:.2f} " + data_e2e)
    # One plot file for each algorithm.
    with open(os.path.join(DIR_TO_SAVE, f"{algo_setting}.dat"), 'w') as f:
        f.write("".join(comps))
        f.write("".join(e2es))