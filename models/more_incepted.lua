local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolution
local Tanh        = nn.Tanh
local ReLU        = nn.ReLU
local Max         = nn.SpatialMaxPooling
local View        = nn.View
local Linear      = nn.Linear
local Dropout     = nn.Dropout

local conv5  = nn.Sequential()
local conv3  = nn.Sequential()
local concat = nn.DepthConcat(2)
local conv5_2 = nn.Sequential()
local conv3_2 = nn.Sequential()
local concat_2 = nn.DepthConcat(2)
local model  = nn.Sequential()


conv5:add(Convolution(3, 8, 1, 1)) -- 8*32*32
conv5:add(ReLU())
conv5:add(Convolution(8, 32, 5, 5, 1, 1, 2, 2))  -- 32*32*32
conv5:add(ReLU())

conv3:add(Convolution(3, 8, 1, 1)) -- 8*32*32
conv3:add(ReLU())
conv3:add(Convolution(8, 32, 3, 3, 1, 1, 1, 1)) -- 32*32*32
conv3:add(ReLU())

concat:add(conv5) -- 32*32*32
concat:add(conv3) -- 64*32*32

model:add(concat) -- 64*32*32
model:add(Max(3,3,1,1)) -- 64*30*30
model:add(Convolution(64, 512, 5, 5)) -- 1024*26*26
model:add(ReLU())
model:add(Convolution(512, 1024, 3, 3)) -- 4096*24*24
model:add(ReLU())
model:add(Convolution(1024, 2048, 5, 5)) -- 2048*20*20
model:add(ReLU())
model:add(Max(4,4,4,4)) -- 2048*5*5

conv5_2:add(model)
conv5_2:add(Convolution(2048, 2048, 1, 1))
conv5_2:add(ReLU())
conv5_2:add(Convolution(2048, 4096, 5, 5, 1, 1, 2, 2))
conv5_2:add(ReLU())

conv3_2:add(model)
conv3_2:add(Convolution(2048, 2048, 1, 1))
conv3_2:add(ReLU())
conv3_2:add(Convolution(2048, 4096, 3, 3, 1, 1, 1, 1))
conv3_2:add(ReLU())

concat_2:add(conv5_2)
concat_2:add(conv3_2)

model:add(concat_2) -- 8192*5*5
model:add(Max(5, 5, 1, 1)) -- 8192*1*1
model:add(View(8192))
model:add(Linear(8192, 2048))
model:add(Dropout(0.40))
model:add(Linear(2048, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
