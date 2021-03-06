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
model:add(Max(2,2,2,2)) -- 64*16*16
model:add(Convolution(64, 128, 5, 5)) -- 128*12*12
model:add(ReLU())
model:add(Convolution(128, 1024, 5, 5)) -- 1024*8*8
model:add(ReLU())
model:add(Max(4,4,4,4)) -- 1024*2*2
model:add(View(4096))
model:add(Dropout(0.40))
model:add(Linear(4096, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
