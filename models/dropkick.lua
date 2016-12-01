local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = nn.SpatialAveragePooling
local View = nn.View
local Linear = nn.Linear
local Concat = nn.DepthConcat
local Dropout = nn.Dropout

local function inception(input_planes, c1_planes, c3_planes, c5_planes, p_planes, package)

  local Convolution = package[1]
  local ReLU = package[2]
  local Max = package[3]
  local Concat = package[4]

  local concat = Concat(2)
  local conv1 = nn.Sequential()
  local conv3 = nn.Sequential()
  local conv5 = nn.Sequential()
  local pool = nn.Sequential()

  conv1:add(Convolution(input_planes, c1_planes, 1, 1))
  conv1:add(ReLU(true))
  concat:add(conv1)

  conv3:add(Convolution(input_planes, c3_planes[1], 1, 1))
  conv3:add(ReLU(true))
  conv3:add(Convolution(c3_planes[1], c3_planes[2], 3, 3, 1, 1, 1, 1))
  conv3:add(ReLU(true))
  concat:add(conv3)

  conv5:add(Convolution(input_planes, c5_planes[1], 1, 1))
  conv5:add(ReLU(true))
  conv5:add(Convolution(c5_planes[1], c5_planes[2], 5, 5, 1, 1, 2, 2))
  conv5:add(ReLU(true))
  concat:add(conv5)

  pool:add(Max(3, 3, 1, 1, 1, 1))
  pool:add(Convolution(input_planes, p_planes, 1, 1))
  pool:add(ReLU(true))
  concat:add(pool)

  return concat

end


local package = {Convolution, ReLU, Max, Concat}

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5, 1, 1, 1, 1)) -- 3 * 32 * 32 --> 16 * 30 * 30
model:add(ReLU(true))
model:add(Convolution(16, 64, 3, 3)) -- 16 * 30 * 30 --> 64 * 28 * 28
model:add(ReLU(true))
model:add(Max(2,2,2,2)) -- 64 * 28 * 28 --> 64 * 14 * 14
--[[
model:add(Convolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 64 * 14 * 14 --> 128 * 14 * 14
model:add(ReLU())
model:add(Convolution(128, 192, 3, 3)) -- 128 * 12 * 12 --> 256 * 10 * 10
model:add(ReLU())
model:add(Max(2,2,2,2)) -- 256 * 10 * 10 --> 256 * 5 * 5
--]]
model:add(inception(64, 64, {96, 128}, {16, 32}, 32, package)) -- 64 * 14 * 14 --> 256 * 14 * 14
--[[
model:add(inception(256, 128, {128, 192}, {32, 96}, 64, package)) -- 256 * 14 * 14 --> 480 * 14 * 14
model:add(inception(480, 192, {96, 208}, {16, 48}, 64, package)) -- 480 * 14 * 14 --> 512 * 14 * 14
model:add(inception(512, 256, {160, 320}, {32, 128}, 128, package)) -- 512 * 14 * 14 --> 832 * 14 * 14
--]]
model:add(Max(2, 2, 2, 2)) -- 256 * 14 * 14 --> 256 * 7 * 7
model:add(inception(256, 384, {192, 384}, {48, 128}, 128, package)) -- 832 * 7 * 7 --> 1024 * 7 * 7
model:add(Avg(7, 7, 1, 1)) -- 1024 * 7 * 7 --> 1024 * 1 * 1
model:add(View(1024))
model:add(Dropout(0.4))
model:add(Linear(1024, 64))
model:add(ReLU())
model:add(Linear(64, 43))

return model
