import multiprocessing
import subprocess
import operator
import math
import os

def parallel_map(func, elements):
    """
    Aplica de forma paralela la función 'func' en toda la colección 'elements'. Básicamente
    sería: res = [func(x) for x in elements] pero paralelamente.
    """
    cant = len(elements)
    cpus = multiprocessing.cpu_count()
    chunksize = int(math.ceil(cant / cpus))
    outq = multiprocessing.Queue()
    jobs = []
    for i in range(cpus):
        chunk = elements[chunksize * i:chunksize * (i + 1)]
        thread = multiprocessing.Process(
            target=process_chunk, args=(func, chunk, i, outq))
        jobs.append(thread)
        thread.start()

    res = []
    for i in range(cpus):
        res.append(outq.get())

    for j in jobs:
        j.join()

    res.sort(key=operator.itemgetter(0)) #ordeno por índices de chunks
    # for t in res:
        # self.data.extend(t[1])


def process_chunk(func, chunk, chunk_index, outq):
    res = [func(x) for x in chunk]
    outq.put((chunk_index, res))


def parallel_map2(func, elements, params, cores=None):
    cant = len(elements)
    cpus = cores or multiprocessing.cpu_count()
    chunksize = int(math.ceil(cant / cpus))
    jobs = []
    elements_list = list(elements)
    for i in range(cpus):
        chunk = {e: elements[e] for e in elements_list[chunksize * i:chunksize * (i + 1)]}
        thread = multiprocessing.Process(
            target=process_chunk2, args=(func, chunk, i, params))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()


def process_chunk2(func, chunk, chunk_index, params):
    func(chunk, chunk_index, params)


def parallel_map_to_file(func, elements, outfile):
    """
    Aplica de forma paralela la función 'func' en toda la colección 'elements'. Básicamente
    sería: res = [func(x) for x in elements] pero paralelamente.
    """
    cant = len(elements)
    cpus = multiprocessing.cpu_count() * 2
    chunksize = int(math.ceil(cant / cpus))
    jobs = []
    for i in range(cpus):
        chunk = elements[chunksize * i:chunksize * (i + 1)]
        thread = multiprocessing.Process(
            target=process_filechunk, args=(func, chunk, i, outfile))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()
    # unir archivos
    aux_files = ["{}_{}".format(outfile, i) for i in range(cpus)]
    os.system("cat " + " ".join(aux_files) + " > " + outfile)
    for af in aux_files:
        os.remove(af)


def process_filechunk(func, chunk, chunk_index, outfile):
    with open("{}_{}".format(outfile, chunk_index), "w") as out:
        for element in chunk:
            res = func(element)
            if res:
                out.write(res + "\n")
    