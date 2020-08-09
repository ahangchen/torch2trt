import torch.nn.functional as F
import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
import collections


def has_grid_sample_plugin():
    try:
        from torch2trt.plugins import GridSamplerPlugin
        return True
    except:
        return False
    
def get_grid_sample_plugin(size, mode, align_corners):
    from torch2trt.plugins import GridSamplerPlugin
    PLUGIN_NAME = 'grid_sample'
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt'][0]
    torch2trt_plugin = GridSamplerPlugin(size=size, mode=mode, align_corners=align_corners)
    return creator.deserialize_plugin(PLUGIN_NAME, torch2trt_plugin.serializeToString())


@tensorrt_converter('torch.nn.functional.grid_sample', enabled=has_grid_sample_plugin())
def convert_grid_sample_plugin(ctx):
    input = ctx.method_args[0]
    grid = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    grid_trt = trt_(ctx.network, grid)
    output = ctx.method_return

    try:
        mode = get_arg(ctx, 'mode', pos=2, default='bilinear')
    except KeyError:
        mode = 'bilinear'
    
    try:
        padding_mode = get_arg(ctx, 'padding_mode', pos=3, default='zeros')
    except KeyError:
        padding_mode = 'zeros'

    try:
        align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)
    except KeyError:
        align_corners = False

    # currently only works for NCHW
    size = list(output.shape[2:])

    plugin = get_grid_sample_plugin(size=size, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    

    layer = ctx.network.add_plugin_v2([input_trt, grid_trt], plugin)

    output._trt = layer.get_output(0)



class GridSampler(torch.nn.Module):
    def __init__(self, mode, padding_mode, align_corners):
        super(GridSampler, self).__init__()
        self.padding_mode = padding_mode
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        return F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)



