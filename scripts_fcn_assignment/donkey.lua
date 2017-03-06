--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "../cache"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache_assignment2.t7')


-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}
local labelSampleSize = {3, opt.labelSize}

-- read the codebook (40 * 3)

local codebooktxt = '/scratch/16824/3d/list/codebook_40.txt' 
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 

  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end


local div_num, sub_num
div_num = 127.5
sub_num = -1


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.labelSize, opt.labelSize )
   input = input * 255
   return input
end



function makeData_cls(img, label)

  local tlabel = torch.Tensor(opt.batchSize, 3, opt.labelSize, opt.labelSize)
  for i = 1, opt.batchSize do
    tlabel[i] = image.scale(label[i], opt.labelSize, opt.labelSize)
  end
  label = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

  for i = 1, opt.batchSize do 
    for j = 1, opt.labelSize do 
      for k = 1, opt.labelSize do 
        nownorm = torch.reshape(tlabel[{{i},{},{j},{k}}], 3)
        scores = codebook * nownorm
        maxnum, idx = torch.max(scores, 1)
        label[cnt] = idx
      end
    end
  end
  print (label:size())
  -- TODO: the input label is a 3-channel real value image, quantize each pixel into classes (1 ~ 40)
  -- resize the label map from a matrix into a long vector
  -- hint: the label should be a vector with dimension of: opt.batchSize * opt.labelSize * opt.labelSize
  return {img, label}
end


function makeData_cls_pre(img, label)
  tmp = torch.Tensor(img:size()):zero()
  tmp[{{}, 1, {}, {}}] = img[{{}, 3, {}, {}}]
  tmp[{{}, 2, {}, {}}] = img[{{}, 2, {}, {}}]
  tmp[{{}, 3, {}, {}}] = img[{{}, 1, {}, {}}]
  local tlabel = torch.Tensor(opt.batchSize, 3, opt.labelSize, opt.labelSize)
  for i = 1, opt.batchSize do
    tlabel[i] = image.scale(label[i], opt.labelSize, opt.labelSize)
  end
  label = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

  for i = 1, opt.batchSize do 
    for j = 1, opt.labelSize do 
      for k = 1, opt.labelSize do 
        nownorm = torch.reshape(tlabel[{{i},{},{j},{k}}], 3)
        scores = codebook * nownorm
        maxnum, idx = torch.max(scores, 1)
        label[cnt] = idx
      end
    end
  end
  -- TODO: almost same as makeData_cls, need to convert img from RGB to BGR for caffe pre-trained model
  return {tmp, label}
end
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:add( - 127.5 )
   label:div(div_num)
   label:add(sub_num)

   return img, label

end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
   trainLoader.labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()



