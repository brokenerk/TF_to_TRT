ºö
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8Ô
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
r

hl4weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ñd*
shared_name
hl4weigths
k
hl4weigths/Read/ReadVariableOpReadVariableOp
hl4weigths* 
_output_shapes
:
ñd*
dtype0
f
hl4biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name	hl4bias
_
hl4bias/Read/ReadVariableOpReadVariableOphl4bias*
_output_shapes
:d*
dtype0
p

outweigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_name
outweigths
i
outweigths/Read/ReadVariableOpReadVariableOp
outweigths*
_output_shapes

:d*
dtype0
f
outbiasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	outbias
_
outbias/Read/ReadVariableOpReadVariableOpoutbias*
_output_shapes
:*
dtype0

NoOpNoOp
§
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â
valueØBÕ BÎ

h1LW
h1LB
h2LW
h2LB
h3LW
h3LB
h4LW
h4LB
	outW

outB
trainableVar

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
outweigthsoutW/.ATTRIBUTES/VARIABLE_VALUE
<:
VARIABLE_VALUEoutbiasoutB/.ATTRIBUTES/VARIABLE_VALUE
F
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
 

serving_default_xPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿdd
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_x
hl1weigthshl1bias
hl2weigthshl2bias
hl3weigthshl3bias
hl4weigthshl4bias
outweigthsoutbias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_2197
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehl1weigths/Read/ReadVariableOphl1bias/Read/ReadVariableOphl2weigths/Read/ReadVariableOphl2bias/Read/ReadVariableOphl3weigths/Read/ReadVariableOphl3bias/Read/ReadVariableOphl4weigths/Read/ReadVariableOphl4bias/Read/ReadVariableOpoutweigths/Read/ReadVariableOpoutbias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_2426

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
hl1weigthshl1bias
hl2weigthshl2bias
hl3weigthshl3bias
hl4weigthshl4bias
outweigthsoutbias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_2466ù®
¬
ê
"__inference_signature_wrapper_2197
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
	unknown_8
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *"
fR
__inference___call___21702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿdd::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
õ,

__inference___call___2241
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢add_4/ReadVariableOpw
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
Relu_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8 2
Reshape_1/shape
	Reshape_1ReshapeRelu_2:activations:0Reshape_1/shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ñd*
dtype02
MatMul/ReadVariableOp
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
Add_3/ReadVariableOpw
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Add_3U
Relu_3Relu	Add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_3
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_3:activations:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOp{
add_4AddV2MatMul_1:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4Ì
IdentityIdentity	add_4:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿdd::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
Æ

__inference__traced_save_2426
file_prefix)
%savev2_hl1weigths_read_readvariableop&
"savev2_hl1bias_read_readvariableop)
%savev2_hl2weigths_read_readvariableop&
"savev2_hl2bias_read_readvariableop)
%savev2_hl3weigths_read_readvariableop&
"savev2_hl3bias_read_readvariableop)
%savev2_hl4weigths_read_readvariableop&
"savev2_hl4bias_read_readvariableop)
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
ShardedFilenameë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEBh2LW/.ATTRIBUTES/VARIABLE_VALUEBh2LB/.ATTRIBUTES/VARIABLE_VALUEBh3LW/.ATTRIBUTES/VARIABLE_VALUEBh3LB/.ATTRIBUTES/VARIABLE_VALUEBh4LW/.ATTRIBUTES/VARIABLE_VALUEBh4LB/.ATTRIBUTES/VARIABLE_VALUEBoutW/.ATTRIBUTES/VARIABLE_VALUEBoutB/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices»
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_hl1weigths_read_readvariableop"savev2_hl1bias_read_readvariableop%savev2_hl2weigths_read_readvariableop"savev2_hl2bias_read_readvariableop%savev2_hl3weigths_read_readvariableop"savev2_hl3bias_read_readvariableop%savev2_hl4weigths_read_readvariableop"savev2_hl4bias_read_readvariableop%savev2_outweigths_read_readvariableop"savev2_outbias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*
_input_shapesr
p: : : : @:@:@::
ñd:d:d:: 2(
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
::&"
 
_output_shapes
:
ñd: 

_output_shapes
:d:$	 

_output_shapes

:d: 


_output_shapes
::

_output_shapes
: 
´+

__inference___call___2285
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢add_4/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapei
ReshapeReshapexReshape/shape:output:0*
T0*&
_output_shapes
:ldd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:ldd *
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
:ldd 2
AddN
ReluReluAdd:z:0*
T0*&
_output_shapes
:ldd 2
Relu
	MaxPool2dMaxPoolRelu:activations:0*&
_output_shapes
:l22 *
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
:l22@*
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
:l22@2
Add_1T
Relu_1Relu	Add_1:z:0*
T0*&
_output_shapes
:l22@2
Relu_1
MaxPool2d_1MaxPoolRelu_1:activations:0*&
_output_shapes
:l@*
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
:l*
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
:l2
Add_2U
Relu_2Relu	Add_2:z:0*
T0*'
_output_shapes
:l2
Relu_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8 2
Reshape_1/shape|
	Reshape_1ReshapeRelu_2:activations:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
lñ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ñd*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:ld2
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:ld2
Add_3L
Relu_3Relu	Add_3:z:0*
T0*
_output_shapes

:ld2
Relu_3
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_3:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:l2

MatMul_1
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOpr
add_4AddV2MatMul_1:product:0add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:l2
add_4Ã
IdentityIdentity	add_4:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add_4/ReadVariableOp*
T0*
_output_shapes

:l2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ldd::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp:I E
&
_output_shapes
:ldd

_user_specified_namex
ñ*
ä
 __inference__traced_restore_2466
file_prefix
assignvariableop_hl1weigths
assignvariableop_1_hl1bias!
assignvariableop_2_hl2weigths
assignvariableop_3_hl2bias!
assignvariableop_4_hl3weigths
assignvariableop_5_hl3bias!
assignvariableop_6_hl4weigths
assignvariableop_7_hl4bias!
assignvariableop_8_outweigths
assignvariableop_9_outbias
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ñ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEBh2LW/.ATTRIBUTES/VARIABLE_VALUEBh2LB/.ATTRIBUTES/VARIABLE_VALUEBh3LW/.ATTRIBUTES/VARIABLE_VALUEBh3LB/.ATTRIBUTES/VARIABLE_VALUEBh4LW/.ATTRIBUTES/VARIABLE_VALUEBh4LB/.ATTRIBUTES/VARIABLE_VALUEBoutW/.ATTRIBUTES/VARIABLE_VALUEBoutB/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
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
AssignVariableOp_8AssignVariableOpassignvariableop_8_outweigthsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_outbiasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
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
_user_specified_namefile_prefix
´+

__inference___call___2329
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢add_4/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿd   d      2
Reshape/shapei
ReshapeReshapexReshape/shape:output:0*
T0*&
_output_shapes
:dd2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd *
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
:dd 2
AddN
ReluReluAdd:z:0*
T0*&
_output_shapes
:dd 2
Relu
	MaxPool2dMaxPoolRelu:activations:0*&
_output_shapes
:22 *
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
:22@*
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
:22@2
Add_1T
Relu_1Relu	Add_1:z:0*
T0*&
_output_shapes
:22@2
Relu_1
MaxPool2d_1MaxPoolRelu_1:activations:0*&
_output_shapes
:@*
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
:*
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
:2
Add_2U
Relu_2Relu	Add_2:z:0*
T0*'
_output_shapes
:2
Relu_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8 2
Reshape_1/shape|
	Reshape_1ReshapeRelu_2:activations:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
ñ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ñd*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:d2
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:d2
Add_3L
Relu_3Relu	Add_3:z:0*
T0*
_output_shapes

:d2
Relu_3
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_3:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOpr
add_4AddV2MatMul_1:product:0add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_4Ã
IdentityIdentity	add_4:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add_4/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
::dd::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp:I E
&
_output_shapes
:dd

_user_specified_namex
õ,

__inference___call___2170
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢add_4/ReadVariableOpw
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
Relu_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8 2
Reshape_1/shape
	Reshape_1ReshapeRelu_2:activations:0Reshape_1/shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ñd*
dtype02
MatMul/ReadVariableOp
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
Add_3/ReadVariableOpw
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Add_3U
Relu_3Relu	Add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Relu_3
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_3:activations:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOp{
add_4AddV2MatMul_1:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4Ì
IdentityIdentity	add_4:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿdd::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd

_user_specified_namex
¬+

__inference___call___2373
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Add_2/ReadVariableOp¢Add_3/ReadVariableOp¢Conv2D/ReadVariableOp¢Conv2D_1/ReadVariableOp¢Conv2D_2/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢add_4/ReadVariableOpw
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
Relu_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8 2
Reshape_1/shape|
	Reshape_1ReshapeRelu_2:activations:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
ñ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ñd*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:d2
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:d2
Add_3L
Relu_3Relu	Add_3:z:0*
T0*
_output_shapes

:d2
Relu_3
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_3:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOpr
add_4AddV2MatMul_1:product:0add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_4Ã
IdentityIdentity	add_4:z:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Add_2/ReadVariableOp^Add_3/ReadVariableOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^add_4/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:dd::::::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2,
Add_2/ReadVariableOpAdd_2/ReadVariableOp2,
Add_3/ReadVariableOpAdd_3/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp:E A
"
_output_shapes
:dd

_user_specified_namex"±L
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
²
h1LW
h1LB
h2LW
h2LB
h3LW
h3LB
h4LW
h4LB
	outW

outB
trainableVar

signatures
__call__"
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
:
ñd2
hl4weigths
:d2hl4bias
:d2
outweigths
:2outbias
f
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
9"
trackable_list_wrapper
,
serving_default"
signature_map
2
__inference___call___2241
__inference___call___2329
__inference___call___2285
__inference___call___2373
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
ÃBÀ
"__inference_signature_wrapper_2197x"
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
 w
__inference___call___2241Z
	
2¢/
(¢%
# 
xÿÿÿÿÿÿÿÿÿdd
ª "ÿÿÿÿÿÿÿÿÿe
__inference___call___2285H
	
)¢&
¢

xldd
ª "le
__inference___call___2329H
	
)¢&
¢

xdd
ª "a
__inference___call___2373D
	
%¢"
¢

xdd
ª " 
"__inference_signature_wrapper_2197z
	
7¢4
¢ 
-ª*
(
x# 
xÿÿÿÿÿÿÿÿÿdd"3ª0
.
output_0"
output_0ÿÿÿÿÿÿÿÿÿ