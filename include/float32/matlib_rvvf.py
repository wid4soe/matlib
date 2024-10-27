#
# Created by widyadewi on 10/15/24.
#

import re
import sys
import traceback
from textwrap import *
from matlib_rvvu import Unroller

from dotmap import DotMap
from enum import Enum

import clang.cindex
from collections import deque

indents = ''
fusable = ['cwiseabs', 'cwisemin', 'cwisemax', 'cwisemul', 'matneg', 
           'matmulf', 'matsub', 'matadd', 'matcopy'] #, 'matset', 'matsetv']

def get_literal_value(node):
    tokens = list(node.get_tokens())
    if tokens:
        return ''.join([token.spelling for token in tokens])
    return None

def node_id(node):
    return str(id(node))[-7:]

def is_literal(node):
    return node.kind == clang.cindex.CursorKind.INTEGER_LITERAL

def is_call(node):
    return node and node.kind == clang.cindex.CursorKind.CALL_EXPR

def is_rvvu(node):
    spelling = get_literal_value(node)
    return is_call(node) and spelling.startswith('__rvvu__')

def is_fuse(node):
    spelling = get_literal_value(node)
    spelling = re.sub(r'\(.*', '', spelling[8:])
    return is_rvvu(node) and spelling in fusable

def print_node(node, level, parent):
    if is_literal(node) or is_call(parent):
        text = get_literal_value(node) or ''
    else:
        text = node.type.spelling
    label = f'[{node.location.line}:{level}:{node_id(parent)}->{node_id(node)}]'
    print(f'{"  " * level}{label} {str(node.kind)[11:]} {node.spelling} [ {text} ]')

def get_params(node):
    params = []
    for child in node.get_children():
        params.append(get_literal_value(child))
    return [params[0][8:], params[1:-2], (eval(params[-2]), eval(params[-1]))]

def scan_sequence(root):
    sequences = []
    prev_node = None
    prev_level = -1
    prev_parent = None
    prev_params = None
    queue = deque([(root, 0, root)])  # Queue holds tuples of (node, level, parent)
    while queue:
        node, level, parent = queue.popleft()
        if is_rvvu(node):
            try:
                params = get_params(node)
                if (not is_fuse(prev_node) or not prev_level == level or 
                    not prev_parent == parent or not prev_params[2] == params[2]):
                    sequences.append([(node, level, parent, params)])
                elif is_rvvu(prev_node):
                    sequences[-1].append((node, level, parent, params))
                else:
                    print("ERROR")
            except:
                pass
        else:
            params = None
        prev_node, prev_level, prev_parent, prev_params = node, level, parent, params
        for child in node.get_children():
            if not is_call(parent):
                queue.append((child, level + 1, node))
    return sequences

class Type(Enum):
    INPUT = 1
    TEMP = 2
    OUTPUT = 3
    SCALAR = 4

def get_vector(seq, param, type: Type):
    if not seq.vectors.get(param):
        if type == Type.SCALAR:
            no = param
            seq.vectors[param] = no
        else:
            no = Unroller.idx
            seq.vectors[param] = no
            init = f'vfloat32_t vec_{no}; float *ptr_{no} = {param};'
            seq.inits.append(init)
            if type == Type.INPUT:
                load = lambda l, vl : f'vec_{no} = __riscv_vle32_v_f32(ptr_{no} + {l}, {vl});'
                seq.loads.append(load)
        Unroller.idx += 1
    else:
        no = seq.vectors.get(param)
    if type == Type.OUTPUT:
        store = lambda l, vl : f'__riscv_vse32_v_f32(ptr_{no} + {l}, vec_{no}, {vl});'
        seq.stores[no] = store
    return no

def get_ops_param_type(ops):
    types = {
        'matneg':   [Type.INPUT, Type.OUTPUT],
        'cwiseabs': [Type.INPUT, Type.OUTPUT],
        'cwisemin': [Type.INPUT, Type.INPUT, Type.OUTPUT],
        'cwisemax': [Type.INPUT, Type.INPUT, Type.OUTPUT],
        'cwisemul': [Type.INPUT, Type.INPUT, Type.OUTPUT],
        'matmulf':  [Type.INPUT, Type.OUTPUT, Type.SCALAR],
        'matsub':   [Type.INPUT, Type.INPUT, Type.OUTPUT],
        'matadd':   [Type.INPUT, Type.INPUT, Type.OUTPUT],
        'matset':   [Type.OUTPUT, Type.SCALAR], 
        'matsetv':  [Type.OUTPUT, Type.INPUT],
        'matcopy':  [Type.INPUT, Type.OUTPUT], 
    }
    return types.get(ops, None)

def get_ops_lambda(ops, seq, params):
    nos = [seq.vectors.get(param) for param in params]
    lambdas = {
        'matneg':   lambda vl: f'vec_{nos[1]} = __riscv_vfneg_v_f32(vec_{nos[0]}, {vl});',
        'cwiseabs': lambda vl: f'vec_{nos[1]} = __riscv_vfabs_v_f32(vec_{nos[0]}, {vl});',
        'cwisemin': lambda vl: f'vec_{nos[2]} = __riscv_vfmin_vv_f32(vec_{nos[0]}, vec_{nos[1]}, {vl});',
        'cwisemax': lambda vl: f'vec_{nos[2]} = __riscv_vfmax_vv_f32(vec_{nos[0]}, vec_{nos[1]}, {vl});',
        'cwisemul': lambda vl: f'vec_{nos[2]} = __riscv_vfmul_vv_f32(vec_{nos[0]}, vec_{nos[1]}, {vl});',
        'matmulf':  lambda vl: f'vec_{nos[1]} = __riscv_vfmul_vf_f32(vec_{nos[0]}, {nos[2]}, {vl});',
        'matsub':   lambda vl: f'vec_{nos[2]} = __riscv_vfsub_vv_f32(vec_{nos[0]}, vec_{nos[1]}, {vl});',
        'matadd':   lambda vl: f'vec_{nos[2]} = __riscv_vfadd_vv_f32(vec_{nos[0]}, vec_{nos[1]}, {vl});',
        'matset':   lambda vl: f'vec_{nos[0]} = __riscv_vfmv_v_f_f32({nos[1]}, {vl});'
    }
    return lambdas.get(ops, None)

def read_sequence(seq, parts):
    ops, params, shape = parts
    types = get_ops_param_type(ops)
    if types is None: return
    for p, param in enumerate(params):
        get_vector(seq, param, types[p])
    seq.ops.append(get_ops_lambda(ops, seq, params))

def write_sequence(seq, parts):
    n, m = parts[2][0], parts[2][1]
    k = m * n
    l = 0
    for x in seq.inits:
        seq.lines.append(str(x))
    while k > 0:
        vl = min(k, Unroller.vlmax)
        for x in seq.loads:
            seq.lines.append(str(x(l, vl)))
        for x in seq.ops:
            if x is not None:
                seq.lines.append(str(x(vl)))
        for x in seq.stores.values():
            seq.lines.append(str(x(l, vl)))
        k -= vl
        l += vl


if __name__ == "__main__":

    DEBUG = False
    prefix = "__rvvu__"
    input_file = sys.argv[1] or 'example_quadrotor_hovering_rvv.ii 1'
    Unroller.lmul = sys.argv[2] or 1

    index = clang.cindex.Index.create()
    translation_unit = index.parse(input_file)
    fusable_sequences = scan_sequence(translation_unit.cursor)

    sequences = []
    sequence_line_nos = {}
    for s, sequence in enumerate(fusable_sequences):
        seq = DotMap({'vectors': {}, 'inits': [], 'loads': [], 'ops': [], 'stores': {}, 'lines': [], 'last_line_no': -1})
        sequences.append(seq)
        for node, level, parent, parts in sequence:
            sequence_line_nos[node.location.line] = s
            seq.last_line_no = max(seq.last_line_no, node.location.line)
            read_sequence(seq, parts)
        write_sequence(seq, parts)

    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    print('#include "matlib_lmul.h"')

    for line_no, line in enumerate(lines):

        if re.match(r"\# (\d+) .+", line):
            continue

        pattern_void = r"(\s+){}(\w+)\s*\((.+)\)".format(prefix)
        match_void = re.match(pattern_void, line)
        pattern_return = r"(\s+)(.+)\s*=\s*{}(.+?)\s*\((.+)\)(.*)".format(prefix)
        match_return = re.match(pattern_return, line)
        if match_void:
            indents = match_void.group(1)
            method = match_void.group(2)
            arguments = match_void.group(3).strip().split(', ')
            if DEBUG:
                print("indents: ", len(indents))
                print("function: ", method)
                print("arguments: ", arguments)
            try:
                fuse = ''
                if method in fusable:
                    # if line is fused, and sequence not output yet, then output else skip
                    fused_sequence_no = sequence_line_nos[line_no + 1]
                    print(f"{indents}// fseq {fused_sequence_no}: {dedent(line)}", end="")
                    sequence = sequences[fused_sequence_no]
                    if line_no + 1 == sequence.last_line_no:
                        for line in sequence.lines:
                            print(f"{indents}{line}")
                else:
                    print(f"{indents}// line {line_no}: {dedent(line)}", end="")
                    getattr(Unroller, method)(indents, *arguments)
            except Exception as e:
                if DEBUG:
                    print(f'// exception {e}')
                    traceback.print_exc()
                print(f"{indents}{method}_rvv({match_void.group(3)});")
        elif match_return:
            indents = match_return.group(1)
            target = match_return.group(2)
            method = match_return.group(3)
            arguments = match_return.group(4).strip().split(', ')
            tail = match_return.group(5)
            try:
                print(f"{indents}// line {line_no}: {dedent(line)}", end="")
                getattr(Unroller, method)(indents, target, tail, *arguments)
            except Exception as e:
                if DEBUG:
                    print(f'// exception {e}')
                    traceback.print_exc()
                print(f"{indents}{target} = {method}_rvv({match_return.group(4)});")
        else:
            print(line, end="")
