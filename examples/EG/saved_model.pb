½
Ì£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878ý
|
Conv1weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv1weigths
u
 Conv1weigths/Read/ReadVariableOpReadVariableOpConv1weigths*&
_output_shapes
:*
dtype0
j
	Conv1biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	Conv1bias
c
Conv1bias/Read/ReadVariableOpReadVariableOp	Conv1bias*
_output_shapes
:*
dtype0

Conv2_age_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv2_age_weigths

%Conv2_age_weigths/Read/ReadVariableOpReadVariableOpConv2_age_weigths*&
_output_shapes
:*
dtype0
t
Conv2_age_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv2_age_bias
m
"Conv2_age_bias/Read/ReadVariableOpReadVariableOpConv2_age_bias*
_output_shapes
:*
dtype0

Conv3_age_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv3_age_weigths

%Conv3_age_weigths/Read/ReadVariableOpReadVariableOpConv3_age_weigths*&
_output_shapes
:*
dtype0
t
Conv3_age_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv3_age_bias
m
"Conv3_age_bias/Read/ReadVariableOpReadVariableOpConv3_age_bias*
_output_shapes
:*
dtype0

Conv2_gender_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameConv2_gender_weigths

(Conv2_gender_weigths/Read/ReadVariableOpReadVariableOpConv2_gender_weigths*&
_output_shapes
:*
dtype0
z
Conv2_gender_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv2_gender_bias
s
%Conv2_gender_bias/Read/ReadVariableOpReadVariableOpConv2_gender_bias*
_output_shapes
:*
dtype0

Conv3_gender_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameConv3_gender_weigths

(Conv3_gender_weigths/Read/ReadVariableOpReadVariableOpConv3_gender_weigths*&
_output_shapes
:*
dtype0
z
Conv3_gender_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv3_gender_bias
s
%Conv3_gender_bias/Read/ReadVariableOpReadVariableOpConv3_gender_bias*
_output_shapes
:*
dtype0
|
FC1_age_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
2* 
shared_nameFC1_age_weigths
u
#FC1_age_weigths/Read/ReadVariableOpReadVariableOpFC1_age_weigths* 
_output_shapes
:
2*
dtype0
p
FC1_age_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameFC1_age_bias
i
 FC1_age_bias/Read/ReadVariableOpReadVariableOpFC1_age_bias*
_output_shapes
:2*
dtype0
z
FC2_age_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_nameFC2_age_weigths
s
#FC2_age_weigths/Read/ReadVariableOpReadVariableOpFC2_age_weigths*
_output_shapes

:2*
dtype0
p
FC2_age_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameFC2_age_bias
i
 FC2_age_bias/Read/ReadVariableOpReadVariableOpFC2_age_bias*
_output_shapes
:*
dtype0
z
out_age_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameout_age_weigths
s
#out_age_weigths/Read/ReadVariableOpReadVariableOpout_age_weigths*
_output_shapes

:*
dtype0
p
out_age_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameout_age_bias
i
 out_age_bias/Read/ReadVariableOpReadVariableOpout_age_bias*
_output_shapes
:*
dtype0

FC1_gender_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
2*#
shared_nameFC1_gender_weigths
{
&FC1_gender_weigths/Read/ReadVariableOpReadVariableOpFC1_gender_weigths* 
_output_shapes
:
2*
dtype0
v
FC1_gender_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_nameFC1_gender_bias
o
#FC1_gender_bias/Read/ReadVariableOpReadVariableOpFC1_gender_bias*
_output_shapes
:2*
dtype0

FC2_gender_weigthsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_nameFC2_gender_weigths
y
&FC2_gender_weigths/Read/ReadVariableOpReadVariableOpFC2_gender_weigths*
_output_shapes

:2*
dtype0
v
FC2_gender_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameFC2_gender_bias
o
#FC2_gender_bias/Read/ReadVariableOpReadVariableOpFC2_gender_bias*
_output_shapes
:*
dtype0
r
outW_genderVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutW_gender
k
outW_gender/Read/ReadVariableOpReadVariableOpoutW_gender*
_output_shapes

:*
dtype0
n
outB_genderVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutB_gender
g
outB_gender/Read/ReadVariableOpReadVariableOpoutB_gender*
_output_shapes
:*
dtype0

NoOpNoOp
ó
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*®
value¤B¡ B
Ç
h1LW
h1LB
h2LW_age
h2LB_age
h3LW_age
h3LB_age
h2LW_gender
h2LB_gender
	h3LW_gender

h3LB_gender

hFC1LW_age

hFC1LB_age

hFC2LW_age

hFC2LB_age
outW_age
outB_age
hFC1LW_gender
hFC1LB_gender
hFC2LW_gender
hFC2LB_gender
outW_gender
outB_gender
trainableVariables
	ksize
stridesConvo
stridesConvo_2
stridesPooling

signatures
A?
VARIABLE_VALUEConv1weigthsh1LW/.ATTRIBUTES/VARIABLE_VALUE
><
VARIABLE_VALUE	Conv1biash1LB/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEConv2_age_weigths#h2LW_age/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEConv2_age_bias#h2LB_age/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEConv3_age_weigths#h3LW_age/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEConv3_age_bias#h3LB_age/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEConv2_gender_weigths&h2LW_gender/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEConv2_gender_bias&h2LB_gender/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEConv3_gender_weigths&h3LW_gender/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEConv3_gender_bias&h3LB_gender/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEFC1_age_weigths%hFC1LW_age/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEFC1_age_bias%hFC1LB_age/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEFC2_age_weigths%hFC2LW_age/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEFC2_age_bias%hFC2LB_age/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEout_age_weigths#outW_age/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEout_age_bias#outB_age/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEFC1_gender_weigths(hFC1LW_gender/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEFC1_gender_bias(hFC1LB_gender/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEFC2_gender_weigths(hFC2LW_gender/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEFC2_gender_bias(hFC2LB_gender/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEoutW_gender&outW_gender/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEoutB_gender&outB_gender/.ATTRIBUTES/VARIABLE_VALUE
F
0
1
	2

3
4
5
6
7
8
9
 
 
 
 
 

serving_default_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÐÐ
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConv1weigths	Conv1biasConv2_age_weigthsConv2_age_biasConv3_age_weigthsConv3_age_biasFC1_age_weigthsFC1_age_biasFC2_age_weigthsFC2_age_biasout_age_weigthsout_age_biasConv2_gender_weigthsConv2_gender_biasConv3_gender_weigthsConv3_gender_biasFC1_gender_weigthsFC1_gender_biasFC2_gender_weigthsFC2_gender_biasoutW_genderoutB_gender*"
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_3661
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Conv1weigths/Read/ReadVariableOpConv1bias/Read/ReadVariableOp%Conv2_age_weigths/Read/ReadVariableOp"Conv2_age_bias/Read/ReadVariableOp%Conv3_age_weigths/Read/ReadVariableOp"Conv3_age_bias/Read/ReadVariableOp(Conv2_gender_weigths/Read/ReadVariableOp%Conv2_gender_bias/Read/ReadVariableOp(Conv3_gender_weigths/Read/ReadVariableOp%Conv3_gender_bias/Read/ReadVariableOp#FC1_age_weigths/Read/ReadVariableOp FC1_age_bias/Read/ReadVariableOp#FC2_age_weigths/Read/ReadVariableOp FC2_age_bias/Read/ReadVariableOp#out_age_weigths/Read/ReadVariableOp out_age_bias/Read/ReadVariableOp&FC1_gender_weigths/Read/ReadVariableOp#FC1_gender_bias/Read/ReadVariableOp&FC2_gender_weigths/Read/ReadVariableOp#FC2_gender_bias/Read/ReadVariableOpoutW_gender/Read/ReadVariableOpoutB_gender/Read/ReadVariableOpConst*#
Tin
2*
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
__inference__traced_save_4298

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1weigths	Conv1biasConv2_age_weigthsConv2_age_biasConv3_age_weigthsConv3_age_biasConv2_gender_weigthsConv2_gender_biasConv3_gender_weigthsConv3_gender_biasFC1_age_weigthsFC1_age_biasFC2_age_weigthsFC2_age_biasout_age_weigthsout_age_biasFC1_gender_weigthsFC1_gender_biasFC2_gender_weigthsFC2_gender_biasoutW_genderoutB_gender*"
Tin
2*
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
 __inference__traced_restore_4374¬
K
á
__inference___call___4208
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
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesn
l:ÐÐ:::::::::::::::::::::::K G
(
_output_shapes
:ÐÐ

_user_specified_namex
2
	
__inference__traced_save_4298
file_prefix+
'savev2_conv1weigths_read_readvariableop(
$savev2_conv1bias_read_readvariableop0
,savev2_conv2_age_weigths_read_readvariableop-
)savev2_conv2_age_bias_read_readvariableop0
,savev2_conv3_age_weigths_read_readvariableop-
)savev2_conv3_age_bias_read_readvariableop3
/savev2_conv2_gender_weigths_read_readvariableop0
,savev2_conv2_gender_bias_read_readvariableop3
/savev2_conv3_gender_weigths_read_readvariableop0
,savev2_conv3_gender_bias_read_readvariableop.
*savev2_fc1_age_weigths_read_readvariableop+
'savev2_fc1_age_bias_read_readvariableop.
*savev2_fc2_age_weigths_read_readvariableop+
'savev2_fc2_age_bias_read_readvariableop.
*savev2_out_age_weigths_read_readvariableop+
'savev2_out_age_bias_read_readvariableop1
-savev2_fc1_gender_weigths_read_readvariableop.
*savev2_fc1_gender_bias_read_readvariableop1
-savev2_fc2_gender_weigths_read_readvariableop.
*savev2_fc2_gender_bias_read_readvariableop*
&savev2_outw_gender_read_readvariableop*
&savev2_outb_gender_read_readvariableop
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_28eedae401ae4335afea29f6a1ffc2d9/part2	
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
ShardedFilenameõ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueýBúBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEB#h2LW_age/.ATTRIBUTES/VARIABLE_VALUEB#h2LB_age/.ATTRIBUTES/VARIABLE_VALUEB#h3LW_age/.ATTRIBUTES/VARIABLE_VALUEB#h3LB_age/.ATTRIBUTES/VARIABLE_VALUEB&h2LW_gender/.ATTRIBUTES/VARIABLE_VALUEB&h2LB_gender/.ATTRIBUTES/VARIABLE_VALUEB&h3LW_gender/.ATTRIBUTES/VARIABLE_VALUEB&h3LB_gender/.ATTRIBUTES/VARIABLE_VALUEB%hFC1LW_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC1LB_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC2LW_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC2LB_age/.ATTRIBUTES/VARIABLE_VALUEB#outW_age/.ATTRIBUTES/VARIABLE_VALUEB#outB_age/.ATTRIBUTES/VARIABLE_VALUEB(hFC1LW_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC1LB_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC2LW_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC2LB_gender/.ATTRIBUTES/VARIABLE_VALUEB&outW_gender/.ATTRIBUTES/VARIABLE_VALUEB&outB_gender/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¶
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1weigths_read_readvariableop$savev2_conv1bias_read_readvariableop,savev2_conv2_age_weigths_read_readvariableop)savev2_conv2_age_bias_read_readvariableop,savev2_conv3_age_weigths_read_readvariableop)savev2_conv3_age_bias_read_readvariableop/savev2_conv2_gender_weigths_read_readvariableop,savev2_conv2_gender_bias_read_readvariableop/savev2_conv3_gender_weigths_read_readvariableop,savev2_conv3_gender_bias_read_readvariableop*savev2_fc1_age_weigths_read_readvariableop'savev2_fc1_age_bias_read_readvariableop*savev2_fc2_age_weigths_read_readvariableop'savev2_fc2_age_bias_read_readvariableop*savev2_out_age_weigths_read_readvariableop'savev2_out_age_bias_read_readvariableop-savev2_fc1_gender_weigths_read_readvariableop*savev2_fc1_gender_bias_read_readvariableop-savev2_fc2_gender_weigths_read_readvariableop*savev2_fc2_gender_bias_read_readvariableop&savev2_outw_gender_read_readvariableop&savev2_outb_gender_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
22
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

identity_1Identity_1:output:0*õ
_input_shapesã
à: :::::::::::
2:2:2::::
2:2:2:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::&"
 
_output_shapes
:
2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
¯
½
"__inference_signature_wrapper_3661	
input
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

unknown_18

unknown_19

unknown_20
identity

identity_1¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *"
fR
__inference___call___36082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿÐÐ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ

_user_specified_nameinput
K
á
__inference___call___4026
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
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesn
l:ÐÐ:::::::::::::::::::::::K G
(
_output_shapes
:ÐÐ

_user_specified_namex
¸N
å
__inference___call___3608	
input"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapex
ReshapeReshapeinputReshape/shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¯
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpz
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ2
AddY
ReluReluAdd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOp³
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOp
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh2
Add_1V
EluElu	Add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh2
Elu 
	MaxPool_1MaxPoolElu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOpµ
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOp
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ442
Add_2]
Relu_1Relu	Add_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ442
Relu_1£
	MaxPool_2MaxPoolRelu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shape
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOp
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpw
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Add_3U
Relu_2Relu	Add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpy
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_4U
Relu_3Relu	Add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOp{
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOp³
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOp
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh2
Add_6Z
Elu_1Elu	Add_6:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿhh2
Elu_1¢
	MaxPool_3MaxPoolElu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOpµ
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOp
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ442
Add_7]
Relu_4Relu	Add_7:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ442
Relu_4£
	MaxPool_4MaxPoolRelu_4:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shape
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpy
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Add_8U
Relu_5Relu	Add_8:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpy
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Add_9U
Relu_6Relu	Add_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOp~
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_10]
IdentityIdentity	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityb

Identity_1Identity
add_10:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿÐÐ:::::::::::::::::::::::X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐÐ

_user_specified_nameinput
K
á
__inference___call___4117
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
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesn
l:ÐÐ:::::::::::::::::::::::K G
(
_output_shapes
:ÐÐ

_user_specified_namex
óK
á
__inference___call___3935
x"
conv2d_readvariableop_resource
add_readvariableop_resource$
 conv2d_1_readvariableop_resource!
add_1_readvariableop_resource$
 conv2d_2_readvariableop_resource!
add_2_readvariableop_resource"
matmul_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1U
CastCastx*

DstT0*

SrcT0*$
_output_shapes
:ÐÐ2
Castw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   Ð   Ð      2
Reshape/shaper
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*{
_input_shapesj
h:ÐÐ:::::::::::::::::::::::G C
$
_output_shapes
:ÐÐ

_user_specified_namex
õY

 __inference__traced_restore_4374
file_prefix!
assignvariableop_conv1weigths 
assignvariableop_1_conv1bias(
$assignvariableop_2_conv2_age_weigths%
!assignvariableop_3_conv2_age_bias(
$assignvariableop_4_conv3_age_weigths%
!assignvariableop_5_conv3_age_bias+
'assignvariableop_6_conv2_gender_weigths(
$assignvariableop_7_conv2_gender_bias+
'assignvariableop_8_conv3_gender_weigths(
$assignvariableop_9_conv3_gender_bias'
#assignvariableop_10_fc1_age_weigths$
 assignvariableop_11_fc1_age_bias'
#assignvariableop_12_fc2_age_weigths$
 assignvariableop_13_fc2_age_bias'
#assignvariableop_14_out_age_weigths$
 assignvariableop_15_out_age_bias*
&assignvariableop_16_fc1_gender_weigths'
#assignvariableop_17_fc1_gender_bias*
&assignvariableop_18_fc2_gender_weigths'
#assignvariableop_19_fc2_gender_bias#
assignvariableop_20_outw_gender#
assignvariableop_21_outb_gender
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueýBúBh1LW/.ATTRIBUTES/VARIABLE_VALUEBh1LB/.ATTRIBUTES/VARIABLE_VALUEB#h2LW_age/.ATTRIBUTES/VARIABLE_VALUEB#h2LB_age/.ATTRIBUTES/VARIABLE_VALUEB#h3LW_age/.ATTRIBUTES/VARIABLE_VALUEB#h3LB_age/.ATTRIBUTES/VARIABLE_VALUEB&h2LW_gender/.ATTRIBUTES/VARIABLE_VALUEB&h2LB_gender/.ATTRIBUTES/VARIABLE_VALUEB&h3LW_gender/.ATTRIBUTES/VARIABLE_VALUEB&h3LB_gender/.ATTRIBUTES/VARIABLE_VALUEB%hFC1LW_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC1LB_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC2LW_age/.ATTRIBUTES/VARIABLE_VALUEB%hFC2LB_age/.ATTRIBUTES/VARIABLE_VALUEB#outW_age/.ATTRIBUTES/VARIABLE_VALUEB#outB_age/.ATTRIBUTES/VARIABLE_VALUEB(hFC1LW_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC1LB_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC2LW_gender/.ATTRIBUTES/VARIABLE_VALUEB(hFC2LB_gender/.ATTRIBUTES/VARIABLE_VALUEB&outW_gender/.ATTRIBUTES/VARIABLE_VALUEB&outB_gender/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¼
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1weigthsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¡
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2_age_weigthsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2_age_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv3_age_weigthsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv3_age_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¬
AssignVariableOp_6AssignVariableOp'assignvariableop_6_conv2_gender_weigthsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7©
AssignVariableOp_7AssignVariableOp$assignvariableop_7_conv2_gender_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¬
AssignVariableOp_8AssignVariableOp'assignvariableop_8_conv3_gender_weigthsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_conv3_gender_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_fc1_age_weigthsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_fc1_age_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_fc2_age_weigthsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_fc2_age_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_out_age_weigthsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_out_age_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_fc1_gender_weigthsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_fc1_gender_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_fc2_gender_weigthsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19«
AssignVariableOp_19AssignVariableOp#assignvariableop_19_fc2_gender_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_outw_genderIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21§
AssignVariableOp_21AssignVariableOpassignvariableop_21_outb_genderIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÂ
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22µ
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
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
K
á
__inference___call___3752
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
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesn
l:ÐÐ:::::::::::::::::::::::K G
(
_output_shapes
:ÐÐ

_user_specified_namex
K
á
__inference___call___3843
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
add_4_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_5_readvariableop_resource$
 conv2d_3_readvariableop_resource!
add_6_readvariableop_resource$
 conv2d_4_readvariableop_resource!
add_7_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_8_readvariableop_resource$
 matmul_4_readvariableop_resource!
add_9_readvariableop_resource$
 matmul_5_readvariableop_resource"
add_10_readvariableop_resource
identity

identity_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿÐ   Ð      2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:ÐÐ2	
Reshape
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DReshape:output:0Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ*
paddingSAME*
strides
2
Conv2D
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
Add/ReadVariableOpq
AddAddConv2D:output:0Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÐÐ2
AddP
ReluReluAdd:z:0*
T0*(
_output_shapes
:ÐÐ2
Relu
MaxPoolMaxPoolRelu:activations:0*&
_output_shapes
:hh*
ksize
*
paddingSAME*
strides
2	
MaxPool
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_1/ReadVariableOpª
Conv2D_1Conv2DMaxPool:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpw
Add_1AddConv2D_1:output:0Add_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_1M
EluElu	Add_1:z:0*
T0*&
_output_shapes
:hh2
Elu
	MaxPool_1MaxPoolElu:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_1
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_2/ReadVariableOp¬
Conv2D_2Conv2DMaxPool_1:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_2
Add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
Add_2/ReadVariableOpw
Add_2AddConv2D_2:output:0Add_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_2T
Relu_1Relu	Add_2:z:0*
T0*&
_output_shapes
:442
Relu_1
	MaxPool_2MaxPoolRelu_1:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_2s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_1/shapez
	Reshape_1ReshapeMaxPool_2:output:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_1
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulReshape_1:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:22
MatMul
Add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_3/ReadVariableOpn
Add_3AddMatMul:product:0Add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_3L
Relu_2Relu	Add_3:z:0*
T0*
_output_shapes

:22
Relu_2
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1MatMulRelu_2:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_1
Add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype02
Add_4/ReadVariableOpp
Add_4AddMatMul_1:product:0Add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_4L
Relu_3Relu	Add_4:z:0*
T0*
_output_shapes

:2
Relu_3
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_2/ReadVariableOp~
MatMul_2MatMulRelu_3:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_2
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpr
add_5AddV2MatMul_2:product:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_3/ReadVariableOpª
Conv2D_3Conv2DMaxPool:output:0Conv2D_3/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh*
paddingSAME*
strides
2

Conv2D_3
Add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype02
Add_6/ReadVariableOpw
Add_6AddConv2D_3:output:0Add_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
Add_6Q
Elu_1Elu	Add_6:z:0*
T0*&
_output_shapes
:hh2
Elu_1
	MaxPool_3MaxPoolElu_1:activations:0*&
_output_shapes
:44*
ksize
*
paddingSAME*
strides
2
	MaxPool_3
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D_4/ReadVariableOp¬
Conv2D_4Conv2DMaxPool_3:output:0Conv2D_4/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingSAME*
strides
2

Conv2D_4
Add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:*
dtype02
Add_7/ReadVariableOpw
Add_7AddConv2D_4:output:0Add_7/ReadVariableOp:value:0*
T0*&
_output_shapes
:442
Add_7T
Relu_4Relu	Add_7:z:0*
T0*&
_output_shapes
:442
Relu_4
	MaxPool_4MaxPoolRelu_4:activations:0*&
_output_shapes
:*
ksize
*
paddingSAME*
strides
2
	MaxPool_4s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿB  2
Reshape_2/shapez
	Reshape_2ReshapeMaxPool_4:output:0Reshape_2/shape:output:0*
T0* 
_output_shapes
:
2
	Reshape_2
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul_3/ReadVariableOp|
MatMul_3MatMulReshape_2:output:0MatMul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:22

MatMul_3
Add_8/ReadVariableOpReadVariableOpadd_8_readvariableop_resource*
_output_shapes
:2*
dtype02
Add_8/ReadVariableOpp
Add_8AddMatMul_3:product:0Add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:22
Add_8L
Relu_5Relu	Add_8:z:0*
T0*
_output_shapes

:22
Relu_5
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul_4/ReadVariableOp~
MatMul_4MatMulRelu_5:activations:0MatMul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_4
Add_9/ReadVariableOpReadVariableOpadd_9_readvariableop_resource*
_output_shapes
:*
dtype02
Add_9/ReadVariableOpp
Add_9AddMatMul_4:product:0Add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Add_9L
Relu_6Relu	Add_9:z:0*
T0*
_output_shapes

:2
Relu_6
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_5/ReadVariableOp~
MatMul_5MatMulRelu_6:activations:0MatMul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2

MatMul_5
add_10/ReadVariableOpReadVariableOpadd_10_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpu
add_10AddV2MatMul_5:product:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10T
IdentityIdentity	add_5:z:0*
T0*
_output_shapes

:2

IdentityY

Identity_1Identity
add_10:z:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapesn
l:ÐÐ:::::::::::::::::::::::K G
(
_output_shapes
:ÐÐ

_user_specified_namex"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ï
serving_defaultÛ
A
input8
serving_default_input:0ÿÿÿÿÿÿÿÿÿÐÐ<
output_00
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:»
ó
h1LW
h1LB
h2LW_age
h2LB_age
h3LW_age
h3LB_age
h2LW_gender
h2LB_gender
	h3LW_gender

h3LB_gender

hFC1LW_age

hFC1LB_age

hFC2LW_age

hFC2LB_age
outW_age
outB_age
hFC1LW_gender
hFC1LB_gender
hFC2LW_gender
hFC2LB_gender
outW_gender
outB_gender
trainableVariables
	ksize
stridesConvo
stridesConvo_2
stridesPooling

signatures
__call__"
_generic_user_object
$:"2Conv1weigths
:2	Conv1bias
):'2Conv2_age_weigths
:2Conv2_age_bias
):'2Conv3_age_weigths
:2Conv3_age_bias
.:,2Conv2_gender_weigths
:2Conv2_gender_bias
.:,2Conv3_gender_weigths
:2Conv3_gender_bias
!:
22FC1_age_weigths
:22FC1_age_bias
:22FC2_age_weigths
:2FC2_age_bias
:2out_age_weigths
:2out_age_bias
&:$
22FC1_gender_weigths
:22FC1_gender_bias
$:"22FC2_gender_weigths
:2FC2_gender_bias
:2outW_gender
:2outB_gender
f
0
1
	2

3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,
serving_default"
signature_map
í2ê
__inference___call___3843
__inference___call___3935
__inference___call___4026
__inference___call___3608
__inference___call___4117
__inference___call___4208
__inference___call___3752ª
¡²
FullArgSpec 
args
jself
jx
jrate
varargs
 
varkw
 
defaults¢
` 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
/B-
"__inference_signature_wrapper_3661input³
__inference___call___3608	
<¢9
2¢/
)&
inputÿÿÿÿÿÿÿÿÿÐÐ
` 
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
__inference___call___3752v	
/¢,
%¢"

xÐÐ
` 
ª "+¢(

0

1
__inference___call___3843v	
/¢,
%¢"

xÐÐ
` 
ª "+¢(

0

1
__inference___call___3935r	
+¢(
!¢

xÐÐ
` 
ª "+¢(

0

1
__inference___call___4026v	
/¢,
%¢"

xÐÐ
` 
ª "+¢(

0

1
__inference___call___4117v	
/¢,
%¢"

xÐÐ
` 
ª "+¢(

0

1
__inference___call___4208v	
/¢,
%¢"

xÐÐ
` 
ª "+¢(

0

1ç
"__inference_signature_wrapper_3661À	
A¢>
¢ 
7ª4
2
input)&
inputÿÿÿÿÿÿÿÿÿÐÐ"cª`
.
output_0"
output_0ÿÿÿÿÿÿÿÿÿ
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ