require 'torch'
local tnt = require 'torchnet'
local image = require 'image'
local WIDTH, HEIGHT = 40, 40

function resize(img)
    return image.scale(img, WIDTH, HEIGHT)
end

function shift(img)
    return image.translate(img, torch.random(-5,5), torch.random(-5,5)) 
end

function rotate(img)
    return image.rotate(img, torch.random(-0.3926990816, 0.3926990816)) --pi/8
end

function scale(img)
    return image.scale(img, WIDTH+torch.random(-5,5), HEIGHT+torch.random(-5,5))
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        --[1] = shift,
        --[2] = rotate,
        --[3] = scale,
        [1] = function(x) return image.rgb2yuv(x) end,
        [2] = resize
    }
    return f(inp)
end

function transformTestInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx, DATA_PATH)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = "./data/test_images/" .. string.format("%05d.ppm", r[1])
    return transformTestInput(image.load(file))
end


