ú
­
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8°Õ
x

hl1weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
hl1weigths
q
hl1weigths/Read/ReadVariableOpReadVariableOp
hl1weigths*&
_output_shapes
: *
dtype0
f
hl1biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	hl1bias
_
hl1bias/Read/ReadVariableOpReadVariableOphl1bias*
_output_shapes
: *
dtype0
x

hl2weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_name
hl2weigths
q
hl2weigths/Read/ReadVariableOpReadVariableOp
hl2weigths*&
_output_shapes
: @*
dtype0
f
hl2biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	hl2bias
_
hl2bias/Read/ReadVariableOpReadVariableOphl2bias*
_output_shapes
:@*
dtype0
y

hl3weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
hl3weigths
r
hl3weigths/Read/ReadVariableOpReadVariableOp
hl3weigths*'
_output_shapes
:@*
dtype0
g
hl3biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl3bias
`
hl3bias/Read/ReadVariableOpReadVariableOphl3bias*
_output_shapes	
:*
dtype0
z

hl4weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
hl4weigths
s
hl4weigths/Read/ReadVariableOpReadVariableOp
hl4weigths*(
_output_shapes
:*
dtype0
g
hl4biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl4bias
`
hl4bias/Read/ReadVariableOpReadVariableOphl4bias*
_output_shapes	
:*
dtype0
z

hl5weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
hl5weigths
s
hl5weigths/Read/ReadVariableOpReadVariableOp
hl5weigths*(
_output_shapes
:*
dtype0
g
hl5biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl5bias
`
hl5bias/Read/ReadVariableOpReadVariableOphl5bias*
_output_shapes	
:*
dtype0
~
hl5weigths_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehl5weigths_1
w
 hl5weigths_1/Read/ReadVariableOpReadVariableOphl5weigths_1*(
_output_shapes
:*
dtype0
k
	hl5bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl5bias_1
d
hl5bias_1/Read/ReadVariableOpReadVariableOp	hl5bias_1*
_output_shapes	
:*
dtype0
~
hl5weigths_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehl5weigths_2
w
 hl5weigths_2/Read/ReadVariableOpReadVariableOphl5weigths_2*(
_output_shapes
:*
dtype0
k
	hl5bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl5bias_2
d
hl5bias_2/Read/ReadVariableOpReadVariableOp	hl5bias_2*
_output_shapes	
:*
dtype0
r

hl8weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
hl8weigths
k
hl8weigths/Read/ReadVariableOpReadVariableOp
hl8weigths* 
_output_shapes
:
*
dtype0
g
hl8biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl8bias
`
hl8bias/Read/ReadVariableOpReadVariableOphl8bias*
_output_shapes	
:*
dtype0
r

hl9weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
hl9weigths
k
hl9weigths/Read/ReadVariableOpReadVariableOp
hl9weigths* 
_output_shapes
:
*
dtype0
g
hl9biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	hl9bias
`
hl9bias/Read/ReadVariableOpReadVariableOphl9bias*
_output_shapes	
:*
dtype0
q

outweigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	5*
shared_name
outweigths
j
outweigths/Read/ReadVariableOpReadVariableOp
outweigths*
_output_shapes
:	5*
dtype0
f
outbiasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_name	outbias
_
outbias/Read/ReadVariableOpReadVariableOpoutbias*
_output_shapes
:5*
dtype0

NoOpNoOp
ß
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
ê
h1LW
h1LB
h2LW
h2LB
h3LW
h3LB
h4LW
h4LB
	h5LW

h5LB
h6LW
h6LB
h7LW
h7LB
h8LW
h8LB
h9LW
h9LB
outW
outB
trainableVar

signatures
?=
VARIABLE_VALUE
hl1weigthsh1LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl1biash1LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl2weigthsh2LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl2biash2LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl3weigthsh3LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl3biash3LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl4weigthsh4LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl4biash4LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl5weigthsh5LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl5biash5LB/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEhl5weigths_1h6LW/.ATTRIBUTES/VARIABLE_VALUE
><
VARIABLE_VALUE	hl5bias_1h6LB/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEhl5weigths_2h7LW/.ATTRIBUTES/VARIABLE_VALUE
><
VARIABLE_VALUE	hl5bias_2h7LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl8weigthsh8LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl8biash8LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
hl9weigthsh9LW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEhl9biash9LB/.ATTRIBUTES/VARIABLE_VALUE
?=
VARIABLE_VALUE
outweigthsoutW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEoutbiasoutB/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 

serving_default_xPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿdd
±
StatefulPartitionedCallStatefulPartitionedCallserving_default_x
hl1weigthshl1bias
hl2weigthshl2bias
hl3weigthshl3bias
hl4weigthshl4bias
hl5weigthshl5biashl5weigths_1	hl5bias_1hl5weigths_2	hl5bias_2
hl8weigthshl8bias
hl9weigthshl9bias
outweigthsoutbias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_630861
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehl1weigths/Read/ReadVariableOphl1bias/Read/ReadVariableOphl2weigths/Read/ReadVariableOphl2bias/Read/ReadVariableOphl3weigths/Read/ReadVariableOphl3bias/Read/ReadVariableOphl4weigths/Read/ReadVariableOphl4bias/Read/ReadVariableOphl5weigths/Read/ReadVariableOphl5bias/Read/ReadVariableOp hl5weigths_1/Read/ReadVariableOphl5bias_1/Read/ReadVariableOp hl5weigths_2/Read/ReadVariableOphl5bias_2/Read/ReadVariableOphl8weigths/Read/ReadVariableOphl8bias/Read/ReadVariableOphl9weigths/Read/ReadVariableOphl9bias/Read/ReadVariableOpoutweigths/Read/ReadVariableOpoutbias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_631249

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
hl1weigthshl1bias
hl2weigthshl2bias
hl3weigthshl3bias
hl4weigthshl4bias
hl5weigthshl5biashl5weigths_1	hl5bias_1hl5weigths_2	hl5bias_2
hl8weigthshl8bias
hl9weigthshl9bias
outweigthsoutbias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_631319Ê


$__inference_signature_wrapper_630861
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *$
fR
__inference___call___6308142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
áY
í	
__inference___call___630814
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_3_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_4_readvariableop_resource$
 conv2d_5_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_6_readvariableop_resource!
add_6_readvariableop_resource"
matmul_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_9_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Add_4/ReadVariableOp¢Add_5/ReadVariableOp¢Add_6/ReadVariableOp¢Add_7/ReadVariableOp¢Add_8/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢Conv2D_3/ReadVariableOp¢Conv2D_4/ReadVariableOp¢Conv2D_5/ReadVariableOp¢Conv2D_6/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢add_9/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shaper
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp­
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd *
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype02
Add/ReadVariableOpx
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd 2
AddW
ReluReluAdd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd 2
Relu¡
	MaxPool2dMaxPoolRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
ksize
*
paddingSAME*
strides
2
	MaxPool2d
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D_1/ReadVariableOpµ
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Add_1/ReadVariableOp
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@2
Add_1]
Relu_1Relu	Add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@2
Relu_1§
MaxPool2d_1MaxPoolRelu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
MaxPool2d_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D_2/ReadVariableOp¸
Conv2D_2Conv2DMaxPool2d_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_2/ReadVariableOp
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_2^
Relu_2Relu	Add_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
MaxPool2d_2MaxPoolRelu_2:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_2
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOp¸
Conv2D_3Conv2DMaxPool2d_2:output:0Conv2D_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_3
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_3/ReadVariableOp
Add_3AddConv2D_3:output:0Add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_3^
Relu_3Relu	Add_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
MaxPool2d_3MaxPoolRelu_3:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¸
Conv2D_4Conv2DMaxPool2d_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_4
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_4/ReadVariableOp
Add_4AddConv2D_4:output:0Add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_4^
Relu_4Relu	Add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_4¨
MaxPool2d_4MaxPoolRelu_4:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_4
Conv2D_5/ReadVariableOpReadVariableOp conv2d_5_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_5/ReadVariableOp¸
Conv2D_5Conv2DMaxPool2d_4:output:0Conv2D_5/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_5
Add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_5/ReadVariableOp
Add_5AddConv2D_5:output:0Add_5/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_5^
Relu_5Relu	Add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_5¨
MaxPool2d_5MaxPoolRelu_5:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_5
Conv2D_6/ReadVariableOpReadVariableOp conv2d_6_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_6/ReadVariableOp¸
Conv2D_6Conv2DMaxPool2d_5:output:0Conv2D_6/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_6
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_6/ReadVariableOp
Add_6AddConv2D_6:output:0Add_6/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_6^
Relu_6Relu	Add_6:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_6¨
MaxPool2d_6MaxPoolRelu_6:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_6s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shape
	Reshape_1ReshapeMaxPool2d_6:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_7/ReadVariableOpx
Add_7AddMatMul:product:0Add_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_7V
Relu_7Relu	Add_7:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_7
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_7:activations:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_8/ReadVariableOpz
Add_8AddMatMul_1:product:0Add_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_8V
Relu_8Relu	Add_8:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_8
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	5*
dtype02
MatMul_2/ReadVariableOp
MatMul_2MatMulRelu_8:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52

MatMul_2
add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:5*
dtype02
add_9/ReadVariableOp{
add_9AddV2MatMul_2:product:0add_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52
add_9Á
IdentityIdentity	add_9:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Add_4/ReadVariableOp^Add_5/ReadVariableOp^Add_6/ReadVariableOp^Add_7/ReadVariableOp^Add_8/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^Conv2D_5/ReadVariableOp^Conv2D_6/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add_9/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2,
Add_4/ReadVariableOpAdd_4/ReadVariableOp2,
Add_5/ReadVariableOpAdd_5/ReadVariableOp2,
Add_6/ReadVariableOpAdd_6/ReadVariableOp2,
Add_7/ReadVariableOpAdd_7/ReadVariableOp2,
Add_8/ReadVariableOpAdd_8/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp22
Conv2D_5/ReadVariableOpConv2D_5/ReadVariableOp22
Conv2D_6/ReadVariableOpConv2D_6/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
Ò
î
__inference___call___631166
x_0
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_49
x_50
x_51
x_52"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_3_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_4_readvariableop_resource$
 conv2d_5_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_6_readvariableop_resource!
add_6_readvariableop_resource"
matmul_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_9_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Add_4/ReadVariableOp¢Add_5/ReadVariableOp¢Add_6/ReadVariableOp¢Add_7/ReadVariableOp¢Add_8/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢Conv2D_3/ReadVariableOp¢Conv2D_4/ReadVariableOp¢Conv2D_5/ReadVariableOp¢Conv2D_6/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢add_9/ReadVariableOp
Cast/xPackx_0x_1x_2x_3x_4x_5x_6x_7x_8x_9x_10x_11x_12x_13x_14x_15x_16x_17x_18x_19x_20x_21x_22x_23x_24x_25x_26x_27x_28x_29x_30x_31x_32x_33x_34x_35x_36x_37x_38x_39x_40x_41x_42x_43x_44x_45x_46x_47x_48x_49x_50x_51x_52*
N5*
T0*&
_output_shapes
:5dd2
Cast/xw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapew
ReshapeReshapeCast/x:output:0Reshape/shape:output:0*
T0*&
_output_shapes
:5dd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:5dd *
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype02
Add/ReadVariableOpo
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*&
_output_shapes
:5dd 2
AddN
ReluReluAdd:z:0*
T0*&
_output_shapes
:5dd 2
Relu
	MaxPool2dMaxPoolRelu:activations:0*&
_output_shapes
:522 *
ksize
*
paddingSAME*
strides
2
	MaxPool2d
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D_1/ReadVariableOp¬
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:522@*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:522@2
Add_1T
Relu_1Relu	Add_1:z:0*
T0*&
_output_shapes
:522@2
Relu_1
MaxPool2d_1MaxPoolRelu_1:activations:0*&
_output_shapes
:5@*
ksize
*
paddingSAME*
strides
2
MaxPool2d_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D_2/ReadVariableOp¯
Conv2D_2Conv2DMaxPool2d_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:5*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_2/ReadVariableOpx
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:52
Add_2U
Relu_2Relu	Add_2:z:0*
T0*'
_output_shapes
:52
Relu_2
MaxPool2d_2MaxPoolRelu_2:activations:0*'
_output_shapes
:5*
ksize
*
paddingSAME*
strides
2
MaxPool2d_2
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOp¯
Conv2D_3Conv2DMaxPool2d_2:output:0Conv2D_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:5*
paddingSAME*
strides
2

Conv2D_3
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_3/ReadVariableOpx
Add_3AddConv2D_3:output:0Add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:52
Add_3U
Relu_3Relu	Add_3:z:0*
T0*'
_output_shapes
:52
Relu_3
MaxPool2d_3MaxPoolRelu_3:activations:0*'
_output_shapes
:5*
ksize
*
paddingSAME*
strides
2
MaxPool2d_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¯
Conv2D_4Conv2DMaxPool2d_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:5*
paddingSAME*
strides
2

Conv2D_4
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_4/ReadVariableOpx
Add_4AddConv2D_4:output:0Add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:52
Add_4U
Relu_4Relu	Add_4:z:0*
T0*'
_output_shapes
:52
Relu_4
MaxPool2d_4MaxPoolRelu_4:activations:0*'
_output_shapes
:5*
ksize
*
paddingSAME*
strides
2
MaxPool2d_4
Conv2D_5/ReadVariableOpReadVariableOp conv2d_5_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_5/ReadVariableOp¯
Conv2D_5Conv2DMaxPool2d_4:output:0Conv2D_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:5*
paddingSAME*
strides
2

Conv2D_5
Add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_5/ReadVariableOpx
Add_5AddConv2D_5:output:0Add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:52
Add_5U
Relu_5Relu	Add_5:z:0*
T0*'
_output_shapes
:52
Relu_5
MaxPool2d_5MaxPoolRelu_5:activations:0*'
_output_shapes
:5*
ksize
*
paddingSAME*
strides
2
MaxPool2d_5
Conv2D_6/ReadVariableOpReadVariableOp conv2d_6_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_6/ReadVariableOp¯
Conv2D_6Conv2DMaxPool2d_5:output:0Conv2D_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:5*
paddingSAME*
strides
2

Conv2D_6
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_6/ReadVariableOpx
Add_6AddConv2D_6:output:0Add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:52
Add_6U
Relu_6Relu	Add_6:z:0*
T0*'
_output_shapes
:52
Relu_6
MaxPool2d_6MaxPoolRelu_6:activations:0*'
_output_shapes
:5*
ksize
*
paddingSAME*
strides
2
MaxPool2d_6s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shape{
	Reshape_1ReshapeMaxPool2d_6:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	52
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpw
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	52
MatMul
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_7/ReadVariableOpo
Add_7AddMatMul:product:0Add_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	52
Add_7M
Relu_7Relu	Add_7:z:0*
T0*
_output_shapes
:	52
Relu_7
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_7:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	52

MatMul_1
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_8/ReadVariableOpq
Add_8AddMatMul_1:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	52
Add_8M
Relu_8Relu	Add_8:z:0*
T0*
_output_shapes
:	52
Relu_8
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	5*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_8:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:552

MatMul_2
add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:5*
dtype02
add_9/ReadVariableOpr
add_9AddV2MatMul_2:product:0add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:552
add_9¸
IdentityIdentity	add_9:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Add_4/ReadVariableOp^Add_5/ReadVariableOp^Add_6/ReadVariableOp^Add_7/ReadVariableOp^Add_8/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^Conv2D_5/ReadVariableOp^Conv2D_6/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add_9/ReadVariableOp*
T0*
_output_shapes

:552

Identity"
identityIdentity:output:0*Ë
_input_shapes¹
¶:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd:dd::::::::::::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2,
Add_4/ReadVariableOpAdd_4/ReadVariableOp2,
Add_5/ReadVariableOpAdd_5/ReadVariableOp2,
Add_6/ReadVariableOpAdd_6/ReadVariableOp2,
Add_7/ReadVariableOpAdd_7/ReadVariableOp2,
Add_8/ReadVariableOpAdd_8/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp22
Conv2D_5/ReadVariableOpConv2D_5/ReadVariableOp22
Conv2D_6/ReadVariableOpConv2D_6/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:G C
"
_output_shapes
:dd

_user_specified_namex/0:GC
"
_output_shapes
:dd

_user_specified_namex/1:GC
"
_output_shapes
:dd

_user_specified_namex/2:GC
"
_output_shapes
:dd

_user_specified_namex/3:GC
"
_output_shapes
:dd

_user_specified_namex/4:GC
"
_output_shapes
:dd

_user_specified_namex/5:GC
"
_output_shapes
:dd

_user_specified_namex/6:GC
"
_output_shapes
:dd

_user_specified_namex/7:GC
"
_output_shapes
:dd

_user_specified_namex/8:G	C
"
_output_shapes
:dd

_user_specified_namex/9:H
D
"
_output_shapes
:dd

_user_specified_namex/10:HD
"
_output_shapes
:dd

_user_specified_namex/11:HD
"
_output_shapes
:dd

_user_specified_namex/12:HD
"
_output_shapes
:dd

_user_specified_namex/13:HD
"
_output_shapes
:dd

_user_specified_namex/14:HD
"
_output_shapes
:dd

_user_specified_namex/15:HD
"
_output_shapes
:dd

_user_specified_namex/16:HD
"
_output_shapes
:dd

_user_specified_namex/17:HD
"
_output_shapes
:dd

_user_specified_namex/18:HD
"
_output_shapes
:dd

_user_specified_namex/19:HD
"
_output_shapes
:dd

_user_specified_namex/20:HD
"
_output_shapes
:dd

_user_specified_namex/21:HD
"
_output_shapes
:dd

_user_specified_namex/22:HD
"
_output_shapes
:dd

_user_specified_namex/23:HD
"
_output_shapes
:dd

_user_specified_namex/24:HD
"
_output_shapes
:dd

_user_specified_namex/25:HD
"
_output_shapes
:dd

_user_specified_namex/26:HD
"
_output_shapes
:dd

_user_specified_namex/27:HD
"
_output_shapes
:dd

_user_specified_namex/28:HD
"
_output_shapes
:dd

_user_specified_namex/29:HD
"
_output_shapes
:dd

_user_specified_namex/30:HD
"
_output_shapes
:dd

_user_specified_namex/31:H D
"
_output_shapes
:dd

_user_specified_namex/32:H!D
"
_output_shapes
:dd

_user_specified_namex/33:H"D
"
_output_shapes
:dd

_user_specified_namex/34:H#D
"
_output_shapes
:dd

_user_specified_namex/35:H$D
"
_output_shapes
:dd

_user_specified_namex/36:H%D
"
_output_shapes
:dd

_user_specified_namex/37:H&D
"
_output_shapes
:dd

_user_specified_namex/38:H'D
"
_output_shapes
:dd

_user_specified_namex/39:H(D
"
_output_shapes
:dd

_user_specified_namex/40:H)D
"
_output_shapes
:dd

_user_specified_namex/41:H*D
"
_output_shapes
:dd

_user_specified_namex/42:H+D
"
_output_shapes
:dd

_user_specified_namex/43:H,D
"
_output_shapes
:dd

_user_specified_namex/44:H-D
"
_output_shapes
:dd

_user_specified_namex/45:H.D
"
_output_shapes
:dd

_user_specified_namex/46:H/D
"
_output_shapes
:dd

_user_specified_namex/47:H0D
"
_output_shapes
:dd

_user_specified_namex/48:H1D
"
_output_shapes
:dd

_user_specified_namex/49:H2D
"
_output_shapes
:dd

_user_specified_namex/50:H3D
"
_output_shapes
:dd

_user_specified_namex/51:H4D
"
_output_shapes
:dd

_user_specified_namex/52
áY
í	
__inference___call___631029
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_3_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_4_readvariableop_resource$
 conv2d_5_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_6_readvariableop_resource!
add_6_readvariableop_resource"
matmul_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_9_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Add_4/ReadVariableOp¢Add_5/ReadVariableOp¢Add_6/ReadVariableOp¢Add_7/ReadVariableOp¢Add_8/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢Conv2D_3/ReadVariableOp¢Conv2D_4/ReadVariableOp¢Conv2D_5/ReadVariableOp¢Conv2D_6/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢add_9/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shaper
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp­
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd *
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype02
Add/ReadVariableOpx
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd 2
AddW
ReluReluAdd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd 2
Relu¡
	MaxPool2dMaxPoolRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 *
ksize
*
paddingSAME*
strides
2
	MaxPool2d
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D_1/ReadVariableOpµ
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Add_1/ReadVariableOp
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@2
Add_1]
Relu_1Relu	Add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22@2
Relu_1§
MaxPool2d_1MaxPoolRelu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
MaxPool2d_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D_2/ReadVariableOp¸
Conv2D_2Conv2DMaxPool2d_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_2/ReadVariableOp
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_2^
Relu_2Relu	Add_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
MaxPool2d_2MaxPoolRelu_2:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_2
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOp¸
Conv2D_3Conv2DMaxPool2d_2:output:0Conv2D_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_3
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_3/ReadVariableOp
Add_3AddConv2D_3:output:0Add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_3^
Relu_3Relu	Add_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
MaxPool2d_3MaxPoolRelu_3:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¸
Conv2D_4Conv2DMaxPool2d_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_4
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_4/ReadVariableOp
Add_4AddConv2D_4:output:0Add_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_4^
Relu_4Relu	Add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_4¨
MaxPool2d_4MaxPoolRelu_4:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_4
Conv2D_5/ReadVariableOpReadVariableOp conv2d_5_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_5/ReadVariableOp¸
Conv2D_5Conv2DMaxPool2d_4:output:0Conv2D_5/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_5
Add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_5/ReadVariableOp
Add_5AddConv2D_5:output:0Add_5/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_5^
Relu_5Relu	Add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_5¨
MaxPool2d_5MaxPoolRelu_5:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_5
Conv2D_6/ReadVariableOpReadVariableOp conv2d_6_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_6/ReadVariableOp¸
Conv2D_6Conv2DMaxPool2d_5:output:0Conv2D_6/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2

Conv2D_6
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_6/ReadVariableOp
Add_6AddConv2D_6:output:0Add_6/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_6^
Relu_6Relu	Add_6:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_6¨
MaxPool2d_6MaxPoolRelu_6:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
MaxPool2d_6s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shape
	Reshape_1ReshapeMaxPool2d_6:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_7/ReadVariableOpx
Add_7AddMatMul:product:0Add_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_7V
Relu_7Relu	Add_7:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_7
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_7:activations:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_8/ReadVariableOpz
Add_8AddMatMul_1:product:0Add_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_8V
Relu_8Relu	Add_8:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_8
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	5*
dtype02
MatMul_2/ReadVariableOp
MatMul_2MatMulRelu_8:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52

MatMul_2
add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:5*
dtype02
add_9/ReadVariableOp{
add_9AddV2MatMul_2:product:0add_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52
add_9Á
IdentityIdentity	add_9:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Add_4/ReadVariableOp^Add_5/ReadVariableOp^Add_6/ReadVariableOp^Add_7/ReadVariableOp^Add_8/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^Conv2D_5/ReadVariableOp^Conv2D_6/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add_9/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ52

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2,
Add_4/ReadVariableOpAdd_4/ReadVariableOp2,
Add_5/ReadVariableOpAdd_5/ReadVariableOp2,
Add_6/ReadVariableOpAdd_6/ReadVariableOp2,
Add_7/ReadVariableOpAdd_7/ReadVariableOp2,
Add_8/ReadVariableOpAdd_8/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp22
Conv2D_5/ReadVariableOpConv2D_5/ReadVariableOp22
Conv2D_6/ReadVariableOpConv2D_6/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
ò,
²
__inference__traced_save_631249
file_prefix)
%savev2_hl1weigths_read_readvariableop&
"savev2_hl1bias_read_readvariableop)
%savev2_hl2weigths_read_readvariableop&
"savev2_hl2bias_read_readvariableop)
%savev2_hl3weigths_read_readvariableop&
"savev2_hl3bias_read_readvariableop)
%savev2_hl4weigths_read_readvariableop&
"savev2_hl4bias_read_readvariableop)
%savev2_hl5weigths_read_readvariableop&
"savev2_hl5bias_read_readvariableop+
'savev2_hl5weigths_1_read_readvariableop(
$savev2_hl5bias_1_read_readvariableop+
'savev2_hl5weigths_2_read_readvariableop(
$savev2_hl5bias_2_read_readvariableop)
%savev2_hl8weigths_read_readvariableop&
"savev2_hl8bias_read_readvariableop)
%savev2_hl9weigths_read_readvariableop&
"savev2_hl9bias_read_readvariableop)
%savev2_outweigths_read_readvariableop&
"savev2_outbias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameµ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEBh2LW/.ATTRIBUTES/VARIABLE_VALUEBh2LB/.ATTRIBUTES/VARIABLE_VALUEBh3LW/.ATTRIBUTES/VARIABLE_VALUEBh3LB/.ATTRIBUTES/VARIABLE_VALUEBh4LW/.ATTRIBUTES/VARIABLE_VALUEBh4LB/.ATTRIBUTES/VARIABLE_VALUEBh5LW/.ATTRIBUTES/VARIABLE_VALUEBh5LB/.ATTRIBUTES/VARIABLE_VALUEBh6LW/.ATTRIBUTES/VARIABLE_VALUEBh6LB/.ATTRIBUTES/VARIABLE_VALUEBh7LW/.ATTRIBUTES/VARIABLE_VALUEBh7LB/.ATTRIBUTES/VARIABLE_VALUEBh8LW/.ATTRIBUTES/VARIABLE_VALUEBh8LB/.ATTRIBUTES/VARIABLE_VALUEBh9LW/.ATTRIBUTES/VARIABLE_VALUEBh9LB/.ATTRIBUTES/VARIABLE_VALUEBoutW/.ATTRIBUTES/VARIABLE_VALUEBoutB/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÄ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_hl1weigths_read_readvariableop"savev2_hl1bias_read_readvariableop%savev2_hl2weigths_read_readvariableop"savev2_hl2bias_read_readvariableop%savev2_hl3weigths_read_readvariableop"savev2_hl3bias_read_readvariableop%savev2_hl4weigths_read_readvariableop"savev2_hl4bias_read_readvariableop%savev2_hl5weigths_read_readvariableop"savev2_hl5bias_read_readvariableop'savev2_hl5weigths_1_read_readvariableop$savev2_hl5bias_1_read_readvariableop'savev2_hl5weigths_2_read_readvariableop$savev2_hl5bias_2_read_readvariableop%savev2_hl8weigths_read_readvariableop"savev2_hl8bias_read_readvariableop%savev2_hl9weigths_read_readvariableop"savev2_hl9bias_read_readvariableop%savev2_outweigths_read_readvariableop"savev2_outbias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesô
ñ: : : : @:@:@::::::::::
::
::	5:5: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	5: 

_output_shapes
:5:

_output_shapes
: 
ÞV
í	
__inference___call___630945
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_3_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_4_readvariableop_resource$
 conv2d_5_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_6_readvariableop_resource!
add_6_readvariableop_resource"
matmul_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_9_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Add_4/ReadVariableOp¢Add_5/ReadVariableOp¢Add_6/ReadVariableOp¢Add_7/ReadVariableOp¢Add_8/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢Conv2D_3/ReadVariableOp¢Conv2D_4/ReadVariableOp¢Conv2D_5/ReadVariableOp¢Conv2D_6/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢add_9/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapei
ReshapeReshapexReshape/shape:output:0*
T0*&
_output_shapes
:dd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd *
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype02
Add/ReadVariableOpo
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd 2
AddN
ReluReluAdd:z:0*
T0*&
_output_shapes
:dd 2
Relu
	MaxPool2dMaxPoolRelu:activations:0*&
_output_shapes
:22 *
ksize
*
paddingSAME*
strides
2
	MaxPool2d
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D_1/ReadVariableOp¬
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:22@*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:22@2
Add_1T
Relu_1Relu	Add_1:z:0*
T0*&
_output_shapes
:22@2
Relu_1
MaxPool2d_1MaxPoolRelu_1:activations:0*&
_output_shapes
:@*
ksize
*
paddingSAME*
strides
2
MaxPool2d_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D_2/ReadVariableOp¯
Conv2D_2Conv2DMaxPool2d_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_2/ReadVariableOpx
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
Add_2U
Relu_2Relu	Add_2:z:0*
T0*'
_output_shapes
:2
Relu_2
MaxPool2d_2MaxPoolRelu_2:activations:0*'
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
MaxPool2d_2
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOp¯
Conv2D_3Conv2DMaxPool2d_2:output:0Conv2D_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

Conv2D_3
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_3/ReadVariableOpx
Add_3AddConv2D_3:output:0Add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
Add_3U
Relu_3Relu	Add_3:z:0*
T0*'
_output_shapes
:2
Relu_3
MaxPool2d_3MaxPoolRelu_3:activations:0*'
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
MaxPool2d_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¯
Conv2D_4Conv2DMaxPool2d_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

Conv2D_4
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_4/ReadVariableOpx
Add_4AddConv2D_4:output:0Add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
Add_4U
Relu_4Relu	Add_4:z:0*
T0*'
_output_shapes
:2
Relu_4
MaxPool2d_4MaxPoolRelu_4:activations:0*'
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
MaxPool2d_4
Conv2D_5/ReadVariableOpReadVariableOp conv2d_5_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_5/ReadVariableOp¯
Conv2D_5Conv2DMaxPool2d_4:output:0Conv2D_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

Conv2D_5
Add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_5/ReadVariableOpx
Add_5AddConv2D_5:output:0Add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
Add_5U
Relu_5Relu	Add_5:z:0*
T0*'
_output_shapes
:2
Relu_5
MaxPool2d_5MaxPoolRelu_5:activations:0*'
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
MaxPool2d_5
Conv2D_6/ReadVariableOpReadVariableOp conv2d_6_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D_6/ReadVariableOp¯
Conv2D_6Conv2DMaxPool2d_5:output:0Conv2D_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

Conv2D_6
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_6/ReadVariableOpx
Add_6AddConv2D_6:output:0Add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
Add_6U
Relu_6Relu	Add_6:z:0*
T0*'
_output_shapes
:2
Relu_6
MaxPool2d_6MaxPoolRelu_6:activations:0*'
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
MaxPool2d_6s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shape{
	Reshape_1ReshapeMaxPool2d_6:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpw
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_7/ReadVariableOpo
Add_7AddMatMul:product:0Add_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Add_7M
Relu_7Relu	Add_7:z:0*
T0*
_output_shapes
:	2
Relu_7
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_7:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

MatMul_1
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes	
:*
dtype02
Add_8/ReadVariableOpq
Add_8AddMatMul_1:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Add_8M
Relu_8Relu	Add_8:z:0*
T0*
_output_shapes
:	2
Relu_8
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	5*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_8:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:52

MatMul_2
add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:5*
dtype02
add_9/ReadVariableOpr
add_9AddV2MatMul_2:product:0add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:52
add_9¸
IdentityIdentity	add_9:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Add_4/ReadVariableOp^Add_5/ReadVariableOp^Add_6/ReadVariableOp^Add_7/ReadVariableOp^Add_8/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^Conv2D_5/ReadVariableOp^Conv2D_6/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add_9/ReadVariableOp*
T0*
_output_shapes

:52

Identity"
identityIdentity:output:0*q
_input_shapes`
^:dd::::::::::::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2,
Add_4/ReadVariableOpAdd_4/ReadVariableOp2,
Add_5/ReadVariableOpAdd_5/ReadVariableOp2,
Add_6/ReadVariableOpAdd_6/ReadVariableOp2,
Add_7/ReadVariableOpAdd_7/ReadVariableOp2,
Add_8/ReadVariableOpAdd_8/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp22
Conv2D_5/ReadVariableOpConv2D_5/ReadVariableOp22
Conv2D_6/ReadVariableOpConv2D_6/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:E A
"
_output_shapes
:dd

_user_specified_namex
ËO
£	
"__inference__traced_restore_631319
file_prefix
assignvariableop_hl1weigths
assignvariableop_1_hl1bias!
assignvariableop_2_hl2weigths
assignvariableop_3_hl2bias!
assignvariableop_4_hl3weigths
assignvariableop_5_hl3bias!
assignvariableop_6_hl4weigths
assignvariableop_7_hl4bias!
assignvariableop_8_hl5weigths
assignvariableop_9_hl5bias$
 assignvariableop_10_hl5weigths_1!
assignvariableop_11_hl5bias_1$
 assignvariableop_12_hl5weigths_2!
assignvariableop_13_hl5bias_2"
assignvariableop_14_hl8weigths
assignvariableop_15_hl8bias"
assignvariableop_16_hl9weigths
assignvariableop_17_hl9bias"
assignvariableop_18_outweigths
assignvariableop_19_outbias
identity_21¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9»
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEBh2LW/.ATTRIBUTES/VARIABLE_VALUEBh2LB/.ATTRIBUTES/VARIABLE_VALUEBh3LW/.ATTRIBUTES/VARIABLE_VALUEBh3LB/.ATTRIBUTES/VARIABLE_VALUEBh4LW/.ATTRIBUTES/VARIABLE_VALUEBh4LB/.ATTRIBUTES/VARIABLE_VALUEBh5LW/.ATTRIBUTES/VARIABLE_VALUEBh5LB/.ATTRIBUTES/VARIABLE_VALUEBh6LW/.ATTRIBUTES/VARIABLE_VALUEBh6LB/.ATTRIBUTES/VARIABLE_VALUEBh7LW/.ATTRIBUTES/VARIABLE_VALUEBh7LB/.ATTRIBUTES/VARIABLE_VALUEBh8LW/.ATTRIBUTES/VARIABLE_VALUEBh8LB/.ATTRIBUTES/VARIABLE_VALUEBh9LW/.ATTRIBUTES/VARIABLE_VALUEBh9LB/.ATTRIBUTES/VARIABLE_VALUEBoutW/.ATTRIBUTES/VARIABLE_VALUEBoutB/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_hl1weigthsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_hl1biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOpassignvariableop_2_hl2weigthsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_hl2biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOpassignvariableop_4_hl3weigthsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_hl3biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¢
AssignVariableOp_6AssignVariableOpassignvariableop_6_hl4weigthsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_hl4biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¢
AssignVariableOp_8AssignVariableOpassignvariableop_8_hl5weigthsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_hl5biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_hl5weigths_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¥
AssignVariableOp_11AssignVariableOpassignvariableop_11_hl5bias_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¨
AssignVariableOp_12AssignVariableOp assignvariableop_12_hl5weigths_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¥
AssignVariableOp_13AssignVariableOpassignvariableop_13_hl5bias_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_hl8weigthsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_hl8biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_hl9weigthsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_hl9biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¦
AssignVariableOp_18AssignVariableOpassignvariableop_18_outweigthsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_outbiasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*§
serving_default
7
x2
serving_default_x:0ÿÿÿÿÿÿÿÿÿdd<
output_00
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ5tensorflow/serving/predict:º

h1LW
h1LB
h2LW
h2LB
h3LW
h3LB
h4LW
h4LB
	h5LW

h5LB
h6LW
h6LB
h7LW
h7LB
h8LW
h8LB
h9LW
h9LB
outW
outB
trainableVar

signatures
__call__"
_generic_user_object
$:" 2
hl1weigths
: 2hl1bias
$:" @2
hl2weigths
:@2hl2bias
%:#@2
hl3weigths
:2hl3bias
&:$2
hl4weigths
:2hl4bias
&:$2
hl5weigths
:2hl5bias
&:$2
hl5weigths
:2hl5bias
&:$2
hl5weigths
:2hl5bias
:
2
hl8weigths
:2hl8bias
:
2
hl9weigths
:2hl9bias
:	52
outweigths
:52outbias
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
,
serving_default"
signature_map
ú2÷
__inference___call___631166
__inference___call___631029
__inference___call___630945
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÅBÂ
$__inference_signature_wrapper_630861x"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 m
__inference___call___630945N	
%¢"
¢

xdd
ª "5
__inference___call___631029d	
2¢/
(¢%
# 
xÿÿÿÿÿÿÿÿÿdd
ª "ÿÿÿÿÿÿÿÿÿ5î
__inference___call___631166Î	
¤¢ 
¢


x/0dd

x/1dd

x/2dd

x/3dd

x/4dd

x/5dd

x/6dd

x/7dd

x/8dd

x/9dd

x/10dd

x/11dd

x/12dd

x/13dd

x/14dd

x/15dd

x/16dd

x/17dd

x/18dd

x/19dd

x/20dd

x/21dd

x/22dd

x/23dd

x/24dd

x/25dd

x/26dd

x/27dd

x/28dd

x/29dd

x/30dd

x/31dd

x/32dd

x/33dd

x/34dd

x/35dd

x/36dd

x/37dd

x/38dd

x/39dd

x/40dd

x/41dd

x/42dd

x/43dd

x/44dd

x/45dd

x/46dd

x/47dd

x/48dd

x/49dd

x/50dd

x/51dd

x/52dd
ª "55­
$__inference_signature_wrapper_630861	
7¢4
¢ 
-ª*
(
x# 
xÿÿÿÿÿÿÿÿÿdd"3ª0
.
output_0"
output_0ÿÿÿÿÿÿÿÿÿ5