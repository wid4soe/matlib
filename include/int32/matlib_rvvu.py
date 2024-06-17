#
# Created by widyadewi on 4/27/24.
#

import re
import sys
from textwrap import *

DEBUG = False
prefix = "__rvvu__"
input_file = sys.argv[1]


class Unroller:

    lmul = 1
    sew = 32
    vlen = 512
    vlmax = int(lmul * vlen / sew)
    idx = 1

    @classmethod
    def print(cls, line):
        print(indent(dedent(line), indents))

    @classmethod
    def setlmul(cls, indents, lmul):
        Unroller.lmul = lmul

    @classmethod
    def matadd(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1}, vec_{i+2};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};
                int *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vadd_vv_i32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_i32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matsub(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1}, vec_{i+2};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};
                int *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vsub_vv_i32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_i32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def maxcoeff(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                int max = std::numeric_limits<int>::min();
                vint32m1_t vec_max = __riscv_vmv_s_x_i32m1(max, 1);
                vint32_t vec_{i};
                int *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_max = __riscv_vredmax_vs_i32_i32(vec_{i}, vec_max, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(f"""\
                max = __riscv_vmv_x_s_i32m1_i32(vec_max);
                {target} = max {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def mincoeff(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                int min = std::numeric_limits<int>::max();
                vint32m1_t vec_min = __riscv_vmv_s_x_i32m1(min, 1);
                vint32_t vec_{i};
                int *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_min = __riscv_vredmin_vs_i32_i32(vec_{i}, vec_min, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(f"""\
                min = __riscv_vmv_x_s_i32m1_i32(vec_min);
                {target} = min {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def matnorm(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i};
                int *ptr_{i} = {a};
                size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                vint32m1_t vec_zero_{i} = __riscv_vmv_v_x_i32m1(0, vlmax_{i});
                vint32_t vec_s_{i} = __riscv_vmv_v_x_i32(0, vlmax_{i});\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_s_{i} = __riscv_vmacc_vv_i32(vec_s_{i}, vec_{i}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(f"""\
                vint32m1_t vec_sum_{i} = __riscv_vredsum_vs_i32_i32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                // int sum_{i} = __riscv_vmv_x_s_i32m1_i32(vec_sum_{i});
                int sum_{i} = 0;
                {target} = sqrt(sum_{i}) {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def matneg(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vneg_v_i32(vec_{i}, {vl});
                __riscv_vse32_v_i32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def transpose(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i};
                int *ptr_{i}, *ptr_{i+1};\
                """)
        for j in range(m):
            k = n
            l = 0
            Unroller.print(f"""\
                ptr_{i} = {a} + {j};
                ptr_{i+1} = {b} + {j * n};\
                """)
            while k > 0:
                vl = min(k, Unroller.vlmax)
                k -= vl
                l += vl
                Unroller.print(f"""\
                    vec_{i} = __riscv_vlse32_v_i32(ptr_{i}, sizeof(int) * {m}, {vl});
                    __riscv_vse32_v_i32(ptr_{i+1}, vec_{i}, {vl});
                    ptr_{i} = {a} + {l * m + j};
                    ptr_{i+1} += {vl};
                """)
        Unroller.idx += 2

    @classmethod
    def cwiseabs(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_n_{i}, vec_{i+1};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_n_{i} = __riscv_vneg_v_i32(vec_{i}, {vl});
                vec_{i+1} = __riscv_vmax_vv_i32(vec_{i}, vec_n_{i}, {vl});
                __riscv_vse32_v_i32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def cwisemin(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1}, vec_{i+2};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};
                int *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vmin_vv_i32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_i32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def cwisemax(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1}, vec_{i+2};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};
                int *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vmax_vv_i32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_i32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matmul(cls, indents, a, b, c, n, m, o):
        n = eval(n)
        m = eval(m)
        o = eval(o)
        i = Unroller.idx
        Unroller.print(f"""\
                        vint32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                        int *ptr_{i}; int *ptr_{i+1}; int *ptr_{i+2} = {c};
                        size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                        vint32m1_t vec_sum_{i};
                        vint32m1_t vec_zero_{i} = __riscv_vmv_v_x_i32m1(0, vlmax_{i});\
                        """)
        for I in range(n):
            for J in range(m):
                k = o
                Unroller.print(f"""\
                        ptr_{i} = {a} + {I * o};
                        ptr_{i+1} = {b} + {J * o};
                        vec_s_{i} = __riscv_vmv_v_x_i32(0, vlmax_{i});\
                        """)
                while k > 0:
                    vl = min(k, Unroller.vlmax)
                    Unroller.print(f"""\
                        vec_{i} = __riscv_vle32_v_i32(ptr_{i}, {vl});
                        vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1}, {vl});
                        vec_s_{i} = __riscv_vmacc_vv_i32(vec_s_{i}, vec_{i}, vec_{i+1}, {vl});\
                        """)
                    if k - vl > 0:
                        Unroller.print(f"""\
                        ptr_{i} += {vl};
                        ptr_{i+1} += {vl};\
                        """)
                    k -= vl
                Unroller.print(f"""\
                        vec_sum_{i} = __riscv_vredsum_vs_i32_i32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                        ptr_{i+2}[{I * m + J}] = __riscv_vmv_x_s_i32m1_i32(vec_sum_{i});\
                        """)
        Unroller.idx += 3

    @classmethod
    def matvec(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(f"""\
                        vint32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                        int *ptr_{i}; int *ptr_{i+1}; int *ptr_{i+2} = {c};
                        size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                        vint32m1_t vec_sum_{i};
                        vint32m1_t vec_zero_{i} = __riscv_vmv_v_x_i32m1(0, vlmax_{i});\
                        """)
        for I in range(n):
            k = m
            Unroller.print(f"""\
                    ptr_{i} = {a} + {I * m};
                    ptr_{i+1} = {b};
                    vec_s_{i} = __riscv_vmv_v_x_i32(0, vlmax_{i});\
                    """)
            while k > 0:
                vl = min(k, Unroller.vlmax)
                Unroller.print(f"""\
                    vec_{i} = __riscv_vle32_v_i32(ptr_{i}, {vl});
                    vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1}, {vl});
                    vec_s_{i} = __riscv_vmacc_vv_i32(vec_s_{i}, vec_{i}, vec_{i+1}, {vl});\
                    """)
                if k > vl:
                    Unroller.print(f"""\
                    ptr_{i} += {vl};
                    ptr_{i+1} += {vl};\
                    """)
                k -= vl
            Unroller.print(f"""\
                    vec_sum_{i} = __riscv_vredsum_vs_i32_i32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                    // ptr_{i+2}[{I}] = __riscv_vmv_x_s_i32m1_i32(vec_sum_{i});\
                    ptr_{i+2}[{I}] = 0;\
                    """)
        Unroller.idx += 3

    @classmethod
    def matvec_transpose(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(f"""\
                        vint32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                        int *ptr_{i}; int *ptr_{i+1}; int *ptr_{i+2} = {c};
                        size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                        vint32m1_t vec_sum_{i};
                        vint32m1_t vec_zero_{i} = __riscv_vmv_v_x_i32m1(0, vlmax_{i});\
                        """)
        for I in range(m):
            k = n
            Unroller.print(f"""\
                    ptr_{i} = {a} + {I};
                    ptr_{i+1} = {b};
                    vec_s_{i} = __riscv_vmv_v_x_i32(0, vlmax_{i});\
                    """)
            while k > 0:
                vl = min(k, Unroller.vlmax)
                Unroller.print(f"""\
                    vec_{i} = __riscv_vlse32_v_i32(ptr_{i}, {m} * sizeof(int), {vl});
                    vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1}, {vl});
                    vec_s_{i} = __riscv_vmacc_vv_i32(vec_s_{i}, vec_{i}, vec_{i+1}, {vl});
                    """)
                if k > vl:
                    Unroller.print(f"""\
                    ptr_{i} += {m} * {vl};
                    ptr_{i+1} += {vl};\
                    """)
                k -= vl
            Unroller.print(f"""\
                    vec_sum_{i} = __riscv_vredsum_vs_i32_i32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                    // ptr_{i+2}[{I}] = __riscv_vmv_x_s_i32m1_i32(vec_sum_{i});\
                    ptr_{i+2}[{I}] = 0;\
                    """)
        Unroller.idx += 3

    @classmethod
    def cwisemul(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1}, vec_{i+2};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};
                int *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_i32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vmul_vv_i32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_i32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matmulf(cls, indents, a, b, f, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i}, vec_{i+1};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vmul_vx_i32(vec_{i}, {f}, {vl});
                __riscv_vse32_v_i32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def matset(cls, indents, a, f, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i};
                int *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vmv_v_x_i32({f}, {vl});
                __riscv_vse32_v_i32(ptr_{i} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 1

    @classmethod
    def matsetv(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i};
                int *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32({b} + {l}, {vl});
                __riscv_vse32_v_i32(ptr_{i} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 1

    @classmethod
    def matcopy(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(f"""\
                vint32_t vec_{i};
                int *ptr_{i} = {a};
                int *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(f"""\
                vec_{i} = __riscv_vle32_v_i32(ptr_{i} + {l}, {vl});
                __riscv_vse32_v_i32(ptr_{i+1} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2


if __name__ != "__main__":
    sys.exit()

Unroller.lmul = sys.argv[2]
with open(input_file, 'r') as f_in:
    lines = f_in.readlines()

print('#include "matlib_lmul.h"')

for line in lines:

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
            print(f"// {line}", end="")
            getattr(Unroller, method)(indents, *arguments)
        except:
            print(f"{indents}{method}_rvv({match_void.group(3)});")
    elif match_return:
        indents = match_return.group(1)
        target = match_return.group(2)
        method = match_return.group(3)
        arguments = match_return.group(4).strip().split(', ')
        tail = match_return.group(5)
        try:
            print(f"// {line}", end="")
            getattr(Unroller, method)(indents, target, tail, *arguments)
        except:
            print(f"{indents}{target} = {method}_rvvu({match_return.group(4)});")
    else:
        print(line, end="")


