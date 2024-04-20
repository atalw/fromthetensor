# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad

class Tensor:
  def __init__():
    pass


"""
*** high level tensor ops ***

# backward pass
deepwalk
backward

# properties
shape
dtype

# data handling
numpy

# convenience
numel
element_size
nbytes

# creation helpers
empty
zeros
ones
arange
eye
rand
randn
randint

# movement ops
__getitem__
__setitem__
reshape
expand
permute
flip
shrink
pad
slice
gather
cat
stack
repeat
chunk
squeeze
unsqueeze

# reduce ops
sum
max
min
mean
std
softmax
log_softmax
argmax
avg_pool2d
max_pool2d
conv2d
dot
matmul
cumsum
triu
tril

# unary function
neg
log
log2
exp
exp2
relu
sigmoid
sin
sqrt
rqsrt
cos
tan

# unary math
abs

# unary activation
swish
tanh
sinh
gelu
leakyrelu

# binary
add
sub
mul
div
pow

# ternary
where

# functional
linear
layernorm
batchnorm
dropout
scaled_dot_product_attention
binary_crossentropy
binary_crossentropy_logits
sparse_categorial_crossentropy
"""