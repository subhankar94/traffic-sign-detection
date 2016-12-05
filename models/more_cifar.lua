local nn = require 'nn'
local image = require 'image'

local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model = nn.Sequential()

model:add(Convolution(3,  32, 5, 5)) -- 16*28*28
model:add(ReLU())
model:add(Convolution(32, 128, 3, 3)) -- 128*26*26
model:add(ReLU())
model:add(Max(3,3,1,1)) -- 128*24*24
model:add(Convolution(128, 256, 5, 5)) -- 256*20*20
model:add(ReLU())
model:add(Convolution(256, 512, 3, 3)) -- 512*18*18
model:add(ReLU())
model:add(Convolution(512, 1024, 3, 3)) -- 1024*16*16
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 1024*8*8
model:add(Convolution(1024, 2048, 5, 5)) -- 2048*4*4
model:add(ReLU())
model:add(Convolution(2048, 4096, 3, 3)) -- 4096*2*2
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 4096*1*1
model:add(View(4096))
model:add(Dropout(0.30))
model:add(Linear(4096, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
