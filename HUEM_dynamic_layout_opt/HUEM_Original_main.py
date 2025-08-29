import csv

from utils.setup import *
from utils.tree import *
from utils.config import *
#from offline.states import *
from offline.dis import *
from offline.cdf import *
import numpy as np
import argparse


def run_original(df, df_sample, out, args, k, queries):
    T = len(queries) // args.interval
    tb = TreeBuilder(df, df_sample, config, args, k, out)
    init_states = tb.get_init_states()
    sg = StateGenerator(tb, init_states, args.interval, epsilon=0, load=True)
    curr_state = init_states[0]
    print("cure_state:",curr_state)
    schedule = {"move": [[0, curr_state.path]], "query": []}
    q = 0
    m = 0
    for i in range(T):
        print("--------------interval-------------------", i)
        new_queries = queries[i * args.interval:(i + 1) * args.interval]
        # Query cost for the new batch
        read_curr = []
        for query in new_queries:
            read, pids = curr_state.eval([query])
            q += read
            read_curr.append(read)
            schedule["query"].extend(pids)
        read_curr = np.average(read_curr)
        print("now cost:", read_curr)
    return schedule, q, m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Periodic baseline.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)

    fnames, files, parts, config = setup_perfile(args)
    # queries = get_workload_perfile(config, files, fnames, args)
    # print("#queries: %d" % len(queries))
    total_query = 0
    total_size = 0
    total_movement = 0
    kk = 0
    for i, fname in enumerate(fnames):

        print("-----------------", fname)
        df, df_sample, k = get_data(config, args, parts, files[i], fname)
        N = len(df)
        output_dir = "%s/%s-%s-%d-%d-%s" % (config["ds"], fname, args.q, args.interval, k, args.method)

        queries = new_get_workload(config, files, fname, args)
        print("# queries: %d" % len(queries))

        decisions, q, m = run_original(df, df_sample, output_dir, args, k, queries)
        # print(decisions)
        total_size += N
        total_query += q * N
        total_movement += m * N
        print("[%s] Query: %f, Movement: %f" % (fname, q, m))
        pickle.dump(decisions, open("resources/schedule/static/%s-%s-%s-%d-%s-%d.p" % (
            config["ds"], fname, args.q, k, args.method, args.alpha), "wb"))
    print("[Periodic (%s,%d)] Query: %f, Movement: %f" % (
        args.policy, args.interval, total_query / total_size, total_movement / total_size))
