from tinygrad.runtime.ops_llvm import LLVMDevice, LLVMProgram, LLVMCompiler
import numpy as np
from llvmlite import ir
from tinygrad import Tensor, Device
from tinygrad.device import MallocAllocator
from tinygrad.helpers import flat_mv, colored
from functools import partialmethod
from tinygrad.dtype import dtypes
from tinygrad.renderer.llvmir import const
import time

np.set_printoptions(linewidth=160, suppress=False)
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


def matmul_16():
  N = 4
  na = np.zeros((N, N), dtype=np.float32)
  nb = np.array([
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
  ]).astype(np.float32)
  nc = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
  ]).astype(np.float32)
  # c = [
  #   [1, 1, 1, 1],
  #   [0, 0, 0, 0],
  #   [1, 1, 1, 1],
  #   [0, 0, 0, 0]
  # ]
  comp = (nb @ nc)
  # nb = nb.T.copy()

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

  for row in range(N):
    AMX.ldx(entry, entry.add(xm, const(row*N*4, dtypes.int64)))
    for col in range(N):
      AMX.ldy(entry, entry.add(const((0 << 62) | (col << 56), dtypes.int64), entry.add(ym, const((row*N*4) + (col+0), dtypes.int64))))

    AMX.fma32(entry, const(0, dtypes.int64))

  for i in range(N):
    AMX.stz(entry, entry.add(zm, const((i*4) << 56 | (i*N*4), dtypes.int64)))

  AMX.clr(entry)

  entry.branch(exit._block)
  exit.ret(const(0, dtypes.int64))
  device = LLVMDevice("llvm")
  ir_str = str(module)
  # print(ir_str)
  prog = LLVMProgram(device, "amx", LLVMCompiler(device).compile(ir_str))
  prog(a, b, c, N**2)
  MallocAllocator.copyout(flat_mv(na.data), a)
  return na, comp

def matmul_N():
  # x, y -> 512 bytes each
  # each operation works on 64 bytes
  # some ops have multiple register options (2 registers for us) enabled by a bit
  # each register is 64 bytes
  # z -> 64 registers in grid = 8x8 grid
  # since we are doing fp32, 2x2 subgrid can compute 32-bit mulacc
  # which probably means we a 16x16 grid
  N = 512
  ele_size = np.float32().itemsize # 4
  ele_count = 512 // 8 // ele_size # 16

  # na = np.zeros(256, dtype=np.float32)
  na = np.zeros((N, N), dtype=np.float32)
  nb = np.random.randn(N, N).astype(np.float32)
  nc = np.random.randn(N, N).astype(np.float32)

  comp = (nb.T @ nc)
  # nb = nb.T.copy()

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

  for y in range(0, N, ele_count*2):
    for x in range(0, N, ele_count*2):
      AMX.set(entry)
      for k in range(N):
        xptr = const(x + (k * N), dtypes.int64)
        yptr = const(y + (k * N), dtypes.int64)

        # multiple register load by setting 62nd bit to 1
        # which means 2 registers are loaded at once
        # one register is 64 bytes, so 32 floats are loaded in adjacent registers here
        # so tile size is 32x32 of the larger matrix
        AMX.ldx(entry, entry.add(const(1<<62, dtypes.int64), entry.add(xm, entry.mul(xptr, const(4, dtypes.int64)))))
        # this is not transposed, we maybe should transpose it. test performance
        AMX.ldy(entry, entry.add(const(1<<62, dtypes.int64), entry.add(ym, entry.mul(yptr, const(4, dtypes.int64)))))

        # fma32 does 16x16 float mulaccs to result in 256 values
        # these are stored in z rows in batches of 16 registers (256*32/8/ 64 bytes per register)
        # result of first multiply added to every 4th row -> 0, 4, 8 ...
        # result of second fma32 stored in row 1, 5, 9 ...
        # result of third fma32 stored in row 2, 6, ...
        # result of fourth fma32 stored in row 3, 7, ...
        # --------------------<Z row> <X offset> <Y offset>
        AMX.fma32(entry, const(0<<20 | (0*16*4)<<10 | (0*16*4), dtypes.int64))
        AMX.fma32(entry, const(1<<20 | (1*16*4)<<10 | (0*16*4), dtypes.int64))
        AMX.fma32(entry, const(2<<20 | (0*16*4)<<10 | (1*16*4), dtypes.int64))
        AMX.fma32(entry, const(3<<20 | (1*16*4)<<10 | (1*16*4), dtypes.int64))

      # get current row and move forward by x col
      gptr = const(((y * N) + x) * 4, dtypes.int64)
      zmp = entry.add(zm, gptr)
      # 2 values for j since 2 consecutive registers are stored at once
      for j in range(2):
        for r in range(16):
          # we are storing pairs of registers by doing (1 << 62), so we multiply by 2 here to skip
          z_row = j*2
          # we are working in a 32x32 tile of a larger NxN matrix
          # when j = 0 (ie 1st register), we want to move foward by N indices to reach the start of the next row
          # when j = 1 (ie 2nd register), we want to move forward by (16 + r)N [1st 16 rows were filled in by register 1 fmas]
          # [and that's why final result is transposed]
          ptr = ((j*16)+r)*N * ele_size
          # when j = 0, row = 0, 4, 8, ... 60
          # when j = 1, row = 2, 6, 10 ... 62
          # store 128 bytes (2 reg at once)
          AMX.stz(entry, entry.add(zmp, const(1<<62 | ((r*4+z_row) << 56) | ptr, dtypes.int64)))
      AMX.clr(entry)

  entry.branch(exit._block)
  exit.ret(const(0, dtypes.int64))
  device = LLVMDevice("llvm")
  ir_str = str(module)
  # print(ir_str)
  prog = LLVMProgram(device, "amx", LLVMCompiler(device).compile(ir_str))
  prog(a, b, c, N**2)
  print("done")
  MallocAllocator.copyout(flat_mv(na.data), a)
  np.testing.assert_allclose(na.T, comp, atol=1e-4, rtol=1e-5)

def matmul_LLVM(M, N, K):
  # M, N, K = 1024, 1024, 1024
  size = np.float32().itemsize
  count = 512 // 8 // size

  # na = np.zeros(256, dtype=np.float32)
  na = np.zeros((M, N), dtype=np.float32)
  nb = np.random.randn(M, K).astype(np.float32)
  nc = np.random.randn(K, N).astype(np.float32)

  # comp = (nb.T @ nc)
  # nb = nb.T.copy()

  a = MallocAllocator.alloc(na.size * np.dtype(np.float32).itemsize)
  b = MallocAllocator.alloc(nb.size * np.dtype(np.float32).itemsize)
  c = MallocAllocator.alloc(nc.size * np.dtype(np.float32).itemsize)
  MallocAllocator.copyin(b, flat_mv(nb.data))
  MallocAllocator.copyin(c, flat_mv(nc.data))

  module = ir.Module(name=__file__)
  func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), "amx")
  entry = ir.IRBuilder(func.append_basic_block(name="entry"))
  exit = ir.IRBuilder(func.append_basic_block(name="exit"))
  zm, bm, cm = [entry.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]

  loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
  loop_2 = ir.IRBuilder(func.append_basic_block(name="loop_x"))
  loop_3 = ir.IRBuilder(func.append_basic_block(name="loop_k"))
  loop_3_exit = ir.IRBuilder(func.append_basic_block(name="loop_k_exit"))
  loop_2_exit = ir.IRBuilder(func.append_basic_block(name="loop_x_exit"))
  loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))

  row = loop_1.phi(ir.IntType(64), name="row")
  col = loop_2.phi(ir.IntType(64), name="col")
  k = loop_3.phi(ir.IntType(64), name="k")

  AMX.set(loop_2)

  # stride
  # kN * 4
  xptr = loop_3_exit.mul(const(size, dtypes.int64), loop_3_exit.add(col, loop_3_exit.mul(k, const(N, dtypes.int64))))
  yptr = loop_3_exit.mul(const(size, dtypes.int64), loop_3_exit.add(row, loop_3_exit.mul(k, const(M, dtypes.int64))))

  # double loads load 32 floats (loading into 2 registers, each register is 64 bytes)
  # load c in x reg and b in y reg to avoid final transpose
  AMX.ldx(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(cm, xptr)))
  AMX.ldy(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(bm, yptr)))

  # <Z row> <X offset> <Y offset>
  AMX.fma32(loop_3_exit, const(0<<20 | (0*count*size)<<10 | (0*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(1<<20 | (1*count*size)<<10 | (0*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(2<<20 | (0*count*size)<<10 | (1*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(3<<20 | (1*count*size)<<10 | (1*count*size), dtypes.int64))

  # store
  # gptr = ((row*N) + col) * size
  gptr = loop_2_exit.mul(loop_2_exit.add(loop_2.mul(row, const(N, dtypes.int64)), col), const(size, dtypes.int64))
  zmp = loop_2_exit.add(zm, gptr)
  for j in range(2):
    for r in range(count):
      z_row = j*2
      ptr = ((j*count)+r)*N*size
      AMX.stz(loop_2_exit, loop_2_exit.add(zmp, const(1 << 62 | ((r*size+z_row) << 56) | ptr, dtypes.int64)))
  AMX.clr(loop_2_exit)

  # count*2 since we're doing double loads 
  yp = loop_1_exit.add(row, const(count*2, dtypes.int64))
  xp = loop_2_exit.add(col, const(count*2, dtypes.int64))
  kp = loop_3_exit.add(k, const(1, dtypes.int64))

  row.add_incoming(const(0, dtypes.int64), entry._block)
  col.add_incoming(const(0, dtypes.int64), loop_1._block)
  k.add_incoming(const(0, dtypes.int64), loop_2._block)
  row.add_incoming(yp, loop_1_exit._block)
  col.add_incoming(xp, loop_2_exit._block)
  k.add_incoming(kp, loop_3_exit._block)

  entry.branch(loop_1._block)
  loop_1.branch(loop_2._block)
  loop_2.branch(loop_3._block)
  loop_3.branch(loop_3_exit._block)
  loop_3_exit.cbranch(loop_3_exit.icmp_unsigned("==", kp, const(K, dtypes.int64)), loop_2_exit._block, loop_3._block)
  loop_2_exit.cbranch(loop_2_exit.icmp_unsigned("==", xp, const(M, dtypes.int64)), loop_1_exit._block, loop_2._block)
  loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, const(N, dtypes.int64)), exit._block, loop_1._block)
  exit.ret(const(0, dtypes.int64))

  device = LLVMDevice("llvm")
  ir_str = str(module)
  # print(ir_str)
  prog = LLVMProgram(device, "amx", LLVMCompiler(device).compile(ir_str))
  prog(a, b, c, N**2)
  MallocAllocator.copyout(flat_mv(na.data), a)
  # np.testing.assert_allclose(na, comp, atol=1e-4, rtol=1e-5)
  tm = min([timeit(lambda: prog(a, b, c, N**2)) for _ in range(20)])
  return tm

def matmul_LLVM_transpose(M, N, K):
  assert M % 16 == 0 and M >= 32
  assert N % 16 == 0 and N >= 32

  size = np.float32().itemsize
  count = 512 // 8 // size

  # na = np.zeros(256, dtype=np.float32)
  na = np.zeros((M, N), dtype=np.float32)
  nb = np.random.randn(M, K).astype(np.float32)
  nc = np.random.randn(K, N).astype(np.float32)

  # comp = (nb @ nc)
  nb = nb.T.copy()

  a = MallocAllocator.alloc(na.size * np.dtype(np.float32).itemsize)
  b = MallocAllocator.alloc(nb.size * np.dtype(np.float32).itemsize)
  c = MallocAllocator.alloc(nc.size * np.dtype(np.float32).itemsize)
  MallocAllocator.copyin(b, flat_mv(nb.data))
  MallocAllocator.copyin(c, flat_mv(nc.data))

  module = ir.Module(name=__file__)
  args = [ir.FloatType().as_pointer()]*3
  args.extend([ir.IntType(64)]*3)
  func = ir.Function(module, ir.FunctionType(ir.IntType(64), args), "amx")
  entry = ir.IRBuilder(func.append_basic_block(name="entry"))
  exit = ir.IRBuilder(func.append_basic_block(name="exit"))
  zm, bm, cm = [entry.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]
  Mm, Nm, Km = [func.args[i] for i in range(3, 6)]

  loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
  loop_2 = ir.IRBuilder(func.append_basic_block(name="loop_x"))
  loop_3 = ir.IRBuilder(func.append_basic_block(name="loop_k"))
  loop_3_exit = ir.IRBuilder(func.append_basic_block(name="loop_k_exit"))
  loop_2_exit = ir.IRBuilder(func.append_basic_block(name="loop_x_exit"))
  loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))

  row = loop_1.phi(ir.IntType(64), name="row")
  col = loop_2.phi(ir.IntType(64), name="col")
  k = loop_3.phi(ir.IntType(64), name="k")

  AMX.set(loop_2)

  # stride
  # kN * 4
  xptr = loop_3_exit.mul(const(size, dtypes.int64), loop_3_exit.add(col, loop_3_exit.mul(k, Nm)))
  yptr = loop_3_exit.mul(const(size, dtypes.int64), loop_3_exit.add(row, loop_3_exit.mul(k, Mm)))

  # double loads load 32 floats (loading into 2 registers, each register is 64 bytes)
  # load c in x reg and b in y reg to avoid final transpose
  AMX.ldx(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(cm, xptr)))
  AMX.ldy(loop_3_exit, loop_3_exit.add(const(1<<62, dtypes.int64), loop_3_exit.add(bm, yptr)))

  # <Z row> <X offset> <Y offset>
  AMX.fma32(loop_3_exit, const(0<<20 | (0*count*size)<<10 | (0*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(1<<20 | (1*count*size)<<10 | (0*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(2<<20 | (0*count*size)<<10 | (1*count*size), dtypes.int64))
  AMX.fma32(loop_3_exit, const(3<<20 | (1*count*size)<<10 | (1*count*size), dtypes.int64))

  # store
  # gptr = ((row*N) + col) * size
  gptr = loop_2_exit.mul(loop_2_exit.add(loop_2.mul(row, Nm), col), const(size, dtypes.int64))
  zmp = loop_2_exit.add(zm, gptr)
  for j in range(2):
    for r in range(count):
      z_row = j*2
      ptr = ((j*count)+r)*N*size
      AMX.stz(loop_2_exit, loop_2_exit.add(zmp, const(1 << 62 | ((r*size+z_row) << 56) | ptr, dtypes.int64)))
  AMX.clr(loop_2_exit)

  # count*2 since we're doing double loads 
  yp = loop_1_exit.add(row, const(count*2, dtypes.int64))
  xp = loop_2_exit.add(col, const(count*2, dtypes.int64))
  kp = loop_3_exit.add(k, const(1, dtypes.int64))

  row.add_incoming(const(0, dtypes.int64), entry._block)
  col.add_incoming(const(0, dtypes.int64), loop_1._block)
  k.add_incoming(const(0, dtypes.int64), loop_2._block)
  row.add_incoming(yp, loop_1_exit._block)
  col.add_incoming(xp, loop_2_exit._block)
  k.add_incoming(kp, loop_3_exit._block)

  entry.branch(loop_1._block)
  loop_1.branch(loop_2._block)
  loop_2.branch(loop_3._block)
  loop_3.branch(loop_3_exit._block)
  loop_3_exit.cbranch(loop_3_exit.icmp_unsigned("==", kp, Km), loop_2_exit._block, loop_3._block)
  loop_2_exit.cbranch(loop_2_exit.icmp_unsigned("==", xp, Nm), loop_1_exit._block, loop_2._block)
  loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, Mm), exit._block, loop_1._block)
  exit.ret(const(0, dtypes.int64))

  device = LLVMDevice("llvm")
  ir_str = str(module)
  # print(ir_str)
  prog = LLVMProgram(device, "amx", LLVMCompiler(device).compile(ir_str))
  MallocAllocator.copyout(flat_mv(na.data), a)
  # np.testing.assert_allclose(na, comp, atol=1e-4, rtol=1e-5)
  print("AMX")
  [timeit(lambda: prog(a, b, c, M, N, K)) for _ in range(10)]

def cpu_matmul(N, M, K):
  nb = np.random.randn(M, K).astype(np.float32)
  nc = np.random.randn(K, N).astype(np.float32)
  print("CPU")
  [timeit(lambda: nb@nc) for _ in range(10)]

def metal_matmul(N, M, K):
  nb = Tensor.randn((M, K), dtype=dtypes.float32).to(device=Device.DEFAULT)
  nc = Tensor.randn((K, N), dtype=dtypes.float32).to(device=Device.DEFAULT)
  assert nb.device == "METAL" and nc.device == "METAL", f"{nb.device=} {nc.device=}"
  print("METAL")
  fn = lambda: (nb@nc).numpy()
  [timeit(fn) for _ in range(10)]

M, N, K = 2048, 2048, 2048

def timeit(fxn):
  st = time.perf_counter()
  fxn()
  t = time.perf_counter() - st
  FLOP = 2*N*M*K
  print(f"{FLOP*1e-9/t:9.2f} GFLOP/S")

if __name__ == "__main__":
  cpu_matmul(M, N, K)
  metal_matmul(M, N, K)
  matmul_LLVM_transpose(M, N, K)

  # for i in range(10):
    # print(f"cpu {FLOP/t1/1e9:9.2f} GFLOP/S | LLVM{FLOP/t2/1e9:9.2f} GFLOP/S")
    # fast_slow = f"transpose slower by {colored(f"{(t2-t1)*1e6:9.2f}", 'red')} us" if t2 > t1 else f"non-tanspose faster by {colored(f"{(t1-t2)*1e6:9.2f}", 'green')} us"
    # print(f"without transpose before gemm = {t1*1e6:9.2f} us, with transpose before gemm = {t2*1e6:9.2f} us, {fast_slow}")