from tinygrad.runtime.ops_llvm import LLVMDevice, LLVMProgram, LLVMCompiler
import numpy as np
from llvmlite import ir
from tinygrad.device import MallocAllocator
from tinygrad.helpers import flat_mv
from functools import partialmethod
from tinygrad.dtype import dtypes
from tinygrad.renderer.llvmir import const

np.set_printoptions(linewidth=160)
np.set_printoptions(linewidth=1000, threshold=10000000000, suppress=False)
class AMX:
  @staticmethod
  def nop_op_imm5(op, imm5, builder): builder.asm(ir.FunctionType(ir.VoidType(), []), f".word (0x201000 + ({op} << 5) + {imm5}); amx op {op} imm {imm5}", "", tuple(), True)
  @staticmethod
  def op_gpr(op, builder, gpr): builder.asm(ir.FunctionType(ir.VoidType(), [ir.IntType(64)]), f".word (0x201000 + ({op} << 5) + 0$0 - ((0$0 >> 4) * 6)); amx op {op} reg $0", "r", (gpr,), True)
  set, clr = partialmethod(nop_op_imm5, 17, 0), partialmethod(nop_op_imm5, 17, 1)
  ldx, ldy, stx, sty = partialmethod(op_gpr, 0), partialmethod(op_gpr, 1), partialmethod(op_gpr, 2), partialmethod(op_gpr, 3)
  ldz, stz, ldzi, stzi = partialmethod(op_gpr, 4), partialmethod(op_gpr, 5), partialmethod(op_gpr, 6), partialmethod(op_gpr, 7)
  extrx, extry = partialmethod(op_gpr, 8), partialmethod(op_gpr, 9)
  fma64, fms64, fma32, fms32 = partialmethod(op_gpr, 10), partialmethod(op_gpr, 11), partialmethod(op_gpr, 12), partialmethod(op_gpr, 13)
  mac16, fma16, fms16 = partialmethod(op_gpr, 14), partialmethod(op_gpr, 15), partialmethod(op_gpr, 16)
  vecint, vecfp, matint, matfp, genlut = partialmethod(op_gpr, 18), partialmethod(op_gpr, 19), partialmethod(op_gpr, 20), partialmethod(op_gpr, 21), partialmethod(op_gpr, 22)

N = 16
# na = np.zeros(256, dtype=np.float32)
na = np.zeros((N, N), dtype=np.float32)
nb = np.random.randn(N, N).astype(np.float32)
nc = np.random.randn(N, N).astype(np.float32)


a = MallocAllocator.alloc(na.size * np.dtype(np.float32).itemsize)
b = MallocAllocator.alloc(nb.size * np.dtype(np.float32).itemsize)
c = MallocAllocator.alloc(nc.size * np.dtype(np.float32).itemsize)

MallocAllocator.copyin(b, flat_mv(nb.data))
MallocAllocator.copyin(c, flat_mv(nc.data))

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), "amx")

entry = ir.IRBuilder(func.append_basic_block(name="entry"))
exit = ir.IRBuilder(func.append_basic_block(name="exit"))
zm, xm, ym = [entry.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]

AMX.set(entry)

for i in range(N):
  AMX.ldx(entry, entry.add(xm, const(i*16*4, dtypes.int64)))
  AMX.ldy(entry, entry.add(ym, const(i*16*4, dtypes.int64)))

  AMX.fma32(entry, const(0, dtypes.int64))

for i in range(N):
  AMX.stz(entry, entry.add(zm, const(i*4 << 56 | (i*16*4), dtypes.int64)))

AMX.clr(entry)

entry.branch(exit._block)
exit.ret(const(0, dtypes.int64))

"""
loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
loop_2 = ir.IRBuilder(func.append_basic_block(name="loop_x"))
loop_3 = ir.IRBuilder(func.append_basic_block(name="loop_k"))
loop_3_exit = ir.IRBuilder(func.append_basic_block(name="loop_k_exit"))
loop_2_exit = ir.IRBuilder(func.append_basic_block(name="loop_x_exit"))
loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))

y = loop_1.phi(ir.IntType(64), name="y")
x = loop_2.phi(ir.IntType(64), name="x")
k = loop_3.phi(ir.IntType(64), name="k")

exit = ir.IRBuilder(func.append_basic_block(name="exit"))

AMX.set(loop_2)

# stride
xptr = loop_3_exit.add(x, loop_3_exit.mul(k, const(N, dtypes.int64)))
yptr = loop_3_exit.add(y, loop_3_exit.mul(k, const(N, dtypes.int64)))

AMX.ldx(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(xm, loop_3_exit.mul(const(4, dtypes.int64), xptr))))
AMX.ldy(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(ym, loop_3_exit.mul(const(4, dtypes.int64), yptr))))

AMX.fma32(loop_3_exit, const(0 << 20 | (0*16*4) << 10 | (0*16*4), dtypes.int64))
AMX.fma32(loop_3_exit, const(1 << 20 | (1*16*4) << 10 | (0*16*4), dtypes.int64))
AMX.fma32(loop_3_exit, const(2 << 20 | (0*16*4) << 10 | (1*16*4), dtypes.int64))
AMX.fma32(loop_3_exit, const(3 << 20 | (1*16*4) << 10 | (1*16*4), dtypes.int64))

gptr = loop_2_exit.mul(const(4, dtypes.int64), loop_2_exit.add(x, loop_2_exit.mul(y, const(N, dtypes.int64))))
zmp = loop_2_exit.add(zm, gptr)
for j in range(2):
  for r in range(16):
    z_row = j*2
    ptr = ((j*16)+r)*N
    AMX.stz(loop_2_exit, loop_2_exit.add(zmp, const(1<<62 | ((r*4+z_row)<<56) | ptr*4, dtypes.int64)))
AMX.clr(loop_2_exit)

yp = loop_1_exit.add(y, const(32, dtypes.int64))
xp = loop_2_exit.add(x, const(32, dtypes.int64))
kp = loop_3_exit.add(k, const(1, dtypes.int64))

y.add_incoming(const(0, dtypes.int64), entry._block)
x.add_incoming(const(0, dtypes.int64), loop_1._block)
k.add_incoming(const(0, dtypes.int64), loop_2._block)
y.add_incoming(yp, loop_1_exit._block)
x.add_incoming(xp, loop_2_exit._block)
k.add_incoming(kp, loop_3_exit._block)

entry.branch(loop_1._block)
loop_1.branch(loop_2._block)
loop_2.branch(loop_3._block)
loop_3.branch(loop_3_exit._block)
loop_3_exit.cbranch(loop_3_exit.icmp_unsigned("==", kp, const(N, dtypes.int64)), loop_2_exit._block, loop_3._block)
loop_2_exit.cbranch(loop_2_exit.icmp_unsigned("==", xp, const(N, dtypes.int64)), loop_1_exit._block, loop_2._block)
loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, const(N, dtypes.int64)), exit._block, loop_1._block)
exit.ret(const(0, dtypes.int64))
"""

device = LLVMDevice("llvm")
ir_str = str(module)
print(ir_str)
prog = LLVMProgram(device, "amx", LLVMCompiler(device).compile(ir_str))
prog(a, b, c, N**2)
MallocAllocator.copyout(flat_mv(na.data), a)


comp = (nb.T @ nc).T
np.testing.assert_allclose(na, comp, atol=1e-4, rtol=1e-5)
# print(na)
# print(comp)