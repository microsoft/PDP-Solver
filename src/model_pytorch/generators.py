# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# generators.py : Defines various types of CNF generators for generating real-time CNF instances.

import numpy as np
import argparse
import os, sys

def is_sat(_var_num, iclause_list):
    ## Note: Invoke your SAT solver of choice here for generating labeled data.
    return False

##########################################################################


class CNFGeneratorBase(object):
    "The base class for all CNF generators."

    def __init__(self, min_n, max_n, min_alpha, max_alpha, alpha_resolution=10):
        self._min_n = min_n
        self._max_n = max_n
        self._min_alpha = min_alpha
        self._max_alpha = max_alpha
        self._alpha = min_alpha
        self._alpha_inc = (max_alpha - min_alpha) / alpha_resolution
        self._alpha_resolution = alpha_resolution

    def generate(self):
        "Generates unlabeled CNF instances."
        pass

    def generate_complete(self):
        "Generates labeled CNF instances."
        pass

    def _to_json(self, n, m, graph_map, edge_feature, label):
        return [[n, m], list(((graph_map[0, :] + 1) * edge_feature).astype(int)), list(graph_map[1, :] + 1), label]

    def _to_dimacs(self, n, m, clause_list):
        body = ''

        for clause in clause_list:
            body += (str(clause)[1:-1].replace(',', '')) + ' 0\n'

        return 'p cnf ' + str(n) + ' ' + str(m) + '\n' + body

    def generate_dataset(self, size, output_dimacs_path, json_output, name, sat_only=True):
        
        max_trial = 50

        if not os.path.exists(output_dimacs_path):
            os.makedirs(output_dimacs_path)

        if not os.path.exists(json_output):
             os.makedirs(json_output)

        output_dimacs_path = os.path.join(output_dimacs_path, name)
        json_output = os.path.join(json_output, name)
       
        for j in range(self._alpha_resolution):
            postfix = '_' + str(j) + '_' + str(self._alpha) + '_' + str(self._alpha + self._alpha_inc)

            if not os.path.exists(output_dimacs_path + postfix):
                os.makedirs(output_dimacs_path + postfix)
            
            with open(json_output + postfix + ".json", 'w') as f:
                for i in range(size):
                    flag = False
                    for _ in range(max_trial):
                        n, m, graph_map, edge_feature, _, label, clause_list = self.generate_complete()

                        if (not sat_only) or (label == 1):
                            flag = True
                            break

                    if flag:
                        f.write(str(self._to_json(n, m, graph_map, edge_feature, label)).replace("'", '"') + '\n')

                        dimacs_file_name = 'dimacs_' + str(i) + '_sat=' + str(label) + '.DIMACS'
                        with open(os.path.join(output_dimacs_path + postfix, dimacs_file_name), 'w') as g:
                            g.write(self._to_dimacs(n, m, clause_list) + '\n')

                    sys.stdout.write("Dataset {:2d}/{:2d}: {:.2f} % complete  \r".format(j + 1, self._alpha_resolution, 100*float(i+1) / size))
                    sys.stdout.flush()

            self._alpha += self._alpha_inc


###############################################################################################


class UniformCNFGenerator(CNFGeneratorBase):
    "Implements the uniformly random CNF generator."

    def __init__(self, min_n, max_n, min_k, max_k, min_alpha, max_alpha, alpha_resolution=10):
        
        super(UniformCNFGenerator, self).__init__(min_n, max_n, min_alpha, max_alpha, alpha_resolution)
        self._min_k = min_k
        self._max_k = max_k
  
    def generate(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._min_alpha, self._max_alpha)
        m = int(n * alpha)

        clause_length = [np.random.randint(self._min_k, min(self._max_k, n-1) + 1) for _ in range(m)]
        edge_num = np.sum(clause_length)

        graph_map = np.zeros((2, edge_num), dtype=np.int32)

        ind = 0
        for i in range(m):
            graph_map[0, ind:(ind+clause_length[i])] = np.random.choice(n, clause_length[i], replace=False)
            graph_map[1, ind:(ind+clause_length[i])] = i
            ind += clause_length[i]

        edge_feature = 2.0 * np.random.choice(2, edge_num) - 1

        return n, m, graph_map, edge_feature, None, -1.0

    def generate_complete(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._alpha, self._alpha + self._alpha_inc)
        m = int(n * alpha)
        max_trial = 10

        clause_set = set()
        clause_list = []
        graph_map = np.zeros((2, 0), dtype=np.int32)
        edge_features = np.zeros(0)

        i = -1
        for _ in range(m):
            for _ in range(max_trial):
                clause_length = np.random.randint(self._min_k, min(self._max_k, n-1) + 1)
                literals = np.sort(np.random.choice(n, clause_length, replace=False))
                edge_feature = 2.0 * np.random.choice(2, clause_length) - 1
                iclause = list(((literals + 1) * edge_feature).astype(int))

                if str(iclause) not in clause_set:
                    i += 1
                    break

            clause_set.add(str(iclause))
            clause_list += [iclause]

            graph_map = np.concatenate((graph_map, np.stack((literals, i * np.ones(clause_length, dtype=np.int32)))), 1)
            edge_features = np.concatenate((edge_features, edge_feature))

        label = is_sat(n, clause_list)
        return n, m, graph_map, edge_features, None, label, clause_list


###############################################################################################


class ModularCNFGenerator(CNFGeneratorBase):
    "Implements the modular random CNF generator according to the Community Attachment model (https://www.iiia.csic.es/sites/default/files/aij16.pdf)"

    def __init__(self, k, min_n, max_n, min_q, max_q, min_c, max_c, min_alpha, max_alpha, alpha_resolution=10):
        
        super(ModularCNFGenerator, self).__init__(min_n, max_n, min_alpha, max_alpha, alpha_resolution)
        self._k = k
        self._min_c = min_c
        self._max_c = max_c
        self._min_q = min_q
        self._max_q = max_q

    def generate(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._min_alpha, self._max_alpha)
        m = int(n * alpha)

        q = np.random.uniform(self._min_q, self._max_q)
        c = np.random.randint(self._min_c, self._max_c + 1)
        c = max(1, min(c, int(n / self._k) - 1))
        size = int(n / c)
        community_size = size * np.ones(c, dtype=np.int32)
        community_size[c - 1] += (n - np.sum(community_size))

        p = q + 1.0 / c
        edge_num = m * self._k

        graph_map = np.zeros((2, edge_num), dtype=np.int32)
        index = np.random.permutation(n)
        
        ind = 0
        for i in range(m):
            coin = np.random.uniform()
            if coin <= p: # Pick from the same community
                community = np.random.randint(0, c)
                graph_map[0, ind:(ind + self._k)] = index[np.random.choice(range(size*community, size*community + community_size[community]), self._k, replace=False)]
            else: # Pick from different communities
                if c >= self._k:
                    communities = np.random.choice(c, self._k, replace=False)
                    temp = np.random.uniform(size = self._k)
                    inner_offset = (temp * community_size[communities]).astype(int)
                    graph_map[0, ind:(ind + self._k)] = index[size*communities + inner_offset]
                else:
                    graph_map[0, ind:(ind + self._k)] = np.random.choice(n, self._k, replace=False)

            graph_map[1, ind:(ind+self._k)] = i
            ind += self._k

        edge_feature = 2.0 * np.random.choice(2, edge_num) - 1

        return n, m, graph_map, edge_feature, None, -1.0

    def generate_complete(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._alpha, self._alpha + self._alpha_inc)
        m = int(n * alpha)
        max_trial = 10

        q = np.random.uniform(self._min_q, self._max_q)
        c = np.random.randint(self._min_c, self._max_c + 1)
        c = max(self._k + 1, min(c, int(n / self._k) - 1))
        size = int(n / c)
        community_size = size * np.ones(c, dtype=np.int32)
        community_size[c - 1] += (n - np.sum(community_size))

        p = q + 1.0 / c
        edge_num = m * self._k

        index = np.random.permutation(n)
        clause_set = set()
        clause_list = []
        graph_map = np.zeros((2, 0), dtype=np.int32)
        edge_features = np.zeros(0)

        i = -1
        for _ in range(m):
            for _ in range(max_trial):
                coin = np.random.uniform()
                if coin <= p: # Pick from the same community
                    community = np.random.randint(0, c)
                    literals = np.sort(index[np.random.choice(range(size*community, size*community + community_size[community]), self._k, replace=False)])
                else: # Pick from different communities
                    communities = np.random.choice(c, self._k, replace=False)
                    temp = np.random.uniform(size = self._k)
                    inner_offset = (temp * community_size[communities]).astype(int)
                    literals = np.sort(index[size*communities + inner_offset])

                edge_feature = 2.0 * np.random.choice(2, self._k) - 1
                iclause = list(((literals + 1) * edge_feature).astype(int))

                if str(iclause) not in clause_set:
                    i += 1
                    break

            clause_set.add(str(iclause))
            clause_list += [iclause]

            graph_map = np.concatenate((graph_map, np.stack((literals, i * np.ones(self._k, dtype=np.int32)))), 1)
            edge_features = np.concatenate((edge_features, edge_feature))

        label = is_sat(n, clause_list)
        return n, m, graph_map, edge_features, None, label, clause_list


###############################################################################################


class VariableModularCNFGenerator(CNFGeneratorBase):
    "Implements a variation of the Community Attachment model with variable sized clauses."

    def __init__(self, min_k, max_k, min_n, max_n, min_q, max_q, min_c, max_c, min_alpha, max_alpha, alpha_resolution=10):
        
        super(VariableModularCNFGenerator, self).__init__(min_n, max_n, min_alpha, max_alpha, alpha_resolution)
        self._min_k = min_k
        self._max_k = max_k
        self._min_c = min_c
        self._max_c = max_c
        self._min_q = min_q
        self._max_q = max_q

    def generate(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._min_alpha, self._max_alpha)
        m = int(n * alpha)

        q = np.random.uniform(self._min_q, self._max_q)
        c = np.random.randint(self._min_c, self._max_c + 1)
        c = max(1, min(c, n))
        size = int(n / c)
        community_size = size * np.ones(c, dtype=np.int32)
        community_size[c - 1] += (n - np.sum(community_size))

        p = q + 1.0 / c
        clause_length = [np.random.randint(min(self._min_k, size), min(self._max_k, n-1, size) + 1) for _ in range(m)]
        edge_num = np.sum(clause_length)

        graph_map = np.zeros((2, edge_num), dtype=np.int32)
        index = np.random.permutation(n)
        
        ind = 0
        for i in range(m):
            coin = np.random.uniform()
            if coin <= p: # Pick from the same community
                community = np.random.randint(0, c)
                graph_map[0, ind:(ind + clause_length[i])] = index[np.random.choice(range(size*community, size*community + community_size[community]), clause_length[i], replace=False)]
            else: # Pick from different communities
                if c >= clause_length[i]:
                    communities = np.random.choice(c, clause_length[i], replace=False)
                    temp = np.random.uniform(size = clause_length[i])
                    inner_offset = (temp * community_size[communities]).astype(int)
                    graph_map[0, ind:(ind + clause_length[i])] = index[size*communities + inner_offset]
                else:
                    graph_map[0, ind:(ind + clause_length[i])] = np.random.choice(n, clause_length[i], replace=False)

            graph_map[1, ind:(ind+clause_length[i])] = i
            ind += clause_length[i]

        edge_feature = 2.0 * np.random.choice(2, edge_num) - 1

        return n, m, graph_map, edge_feature, None, -1.0

    def generate_complete(self):
        n = np.random.randint(self._min_n, self._max_n + 1)
        alpha = np.random.uniform(self._alpha, self._alpha + self._alpha_inc)
        m = int(n * alpha)
        max_trial = 10

        q = np.random.uniform(self._min_q, self._max_q)
        c = np.random.randint(self._min_c, self._max_c + 1)
        c = max(self._k + 1, min(c, int(n / self._k) - 1))
        size = int(n / c)
        community_size = size * np.ones(c, dtype=np.int32)
        community_size[c - 1] += (n - np.sum(community_size))

        p = q + 1.0 / c
        edge_num = m * self._k

        index = np.random.permutation(n)
        clause_set = set()
        clause_list = []
        graph_map = np.zeros((2, 0), dtype=np.int32)
        edge_features = np.zeros(0)

        i = -1
        for _ in range(m):
            for _ in range(max_trial):
                clause_length = np.random.randint(min(self._min_k, size), min(self._max_k, n-1, size) + 1)
                coin = np.random.uniform()
                if coin <= p: # Pick from the same community
                    community = np.random.randint(0, c)
                    literals = np.sort(index[np.random.choice(range(size*community, size*community + community_size[community]), clause_length, replace=False)])
                else: # Pick from different communities
                    if c >= clause_length:
                        communities = np.random.choice(c, clause_length, replace=False)
                        temp = np.random.uniform(size = clause_length)
                        inner_offset = (temp * community_size[communities]).astype(int)
                        literals = np.sort(index[size*communities + inner_offset])
                    else:
                        literals = np.random.choice(n, clause_length, replace=False)

                edge_feature = 2.0 * np.random.choice(2, clause_length) - 1
                iclause = list(((literals + 1) * edge_feature).astype(int))

                if str(iclause) not in clause_set:
                    i += 1
                    break

            clause_set.add(str(iclause))
            clause_list += [iclause]

            graph_map = np.concatenate((graph_map, np.stack((literals, i * np.ones(clause_length, dtype=np.int32)))), 1)
            edge_features = np.concatenate((edge_features, edge_feature))

        label = is_sat(n, clause_list)
        return n, m, graph_map, edge_features, None, label, clause_list


###############################################################################################


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', action='store', type=str)
    parser.add_argument('out_json', action='store', type=str)
    parser.add_argument('name', action='store', type=str)
    parser.add_argument('size', action='store', type=int)
    parser.add_argument('method', action='store', type=str)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

    parser.add_argument('--min_c', action='store', dest='min_c', type=int, default=10)
    parser.add_argument('--max_c', action='store', dest='max_c', type=int, default=40)

    parser.add_argument('--min_q', action='store', dest='min_q', type=float, default=0.3)
    parser.add_argument('--max_q', action='store', dest='max_q', type=float, default=0.9)

    parser.add_argument('--min_k', action='store', dest='min_k', type=int, default=3)
    parser.add_argument('--max_k', action='store', dest='max_k', type=int, default=5)

    parser.add_argument('--min_a', action='store', dest='min_a', type=float, default=2)
    parser.add_argument('--max_a', action='store', dest='max_a', type=float, default=10)
    parser.add_argument('--res', action='store', dest='res', type=int, default=5)

    parser.add_argument('-s', '--sat_only', help='Include SAT examples only', required=False, action='store_true', default=False)

    args = vars(parser.parse_args())

    if args['method'] == 'modular':
        generator = ModularCNFGenerator(k=args['min_k'], min_n=args['min_n'], max_n=args['max_n'], min_q=args['min_q'], 
                max_q=args['max_q'], min_c=args['min_c'], max_c=args['max_c'], min_alpha=args['min_a'], max_alpha=args['max_a'], alpha_resolution=args['res'])
    elif args['method'] == 'v-modular':
        generator = VariableModularCNFGenerator(min_k=args['min_k'], max_k=args['max_k'], min_n=args['min_n'], max_n=args['max_n'], min_q=args['min_q'], 
                max_q=args['max_q'], min_c=args['min_c'], max_c=args['max_c'], min_alpha=args['min_a'], max_alpha=args['max_a'], alpha_resolution=args['res'])
    else:
        generator = UniformCNFGenerator(min_n=args['min_n'], max_n=args['max_n'], min_k=args['min_k'], 
                max_k=args['max_k'], min_alpha=args['min_a'], max_alpha=args['max_a'], alpha_resolution=args['res'])
    
    generator.generate_dataset(args['size'], args['out_dir'], args['out_json'], args['name'], args['sat_only'])