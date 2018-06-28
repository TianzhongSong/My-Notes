from utils.quantize import google_quantize, linear_quantize
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches


def plot_histograms(name, weight):
    fig, ax = plt.subplots()
    n, bins = np.histogram(weight, 50)
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    barpath = path.Path.make_compound_path_from_polys(XY)

    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    # plt.savefig('./histograms/{}.png'.format(name.split('_')[0]))
    plt.show()


if __name__ == "__main__":
    model =  h5py.File('LeNet.h5', 'r')
    keras_version = model.attrs['keras_version']
    backend =model.attrs['backend']
    model_config = model.attrs['model_config']
    print('keras version:{}'.format(keras_version))
    print('backend:{}'.format(backend))
    # print('model config:{}'.format(model_config))
    f = model['model_weights'] 
    qmodel = h5py.File('QLeNet_fp16.h5', mode='w')
    qmodel.attrs['keras_version'] = model.attrs['keras_version']
    qmodel.attrs['backend'] = model.attrs['backend']
    qmodel.attrs['model_config'] = model.attrs['model_config']
    model_weights_group = qmodel.create_group('model_weights')
    model_weights_group.attrs['layer_names'] = [name for name in f.attrs['layer_names']]
    model_weights_group.attrs['backend'] = model.attrs['backend']
    model_weights_group.attrs['model_config'] = model.attrs['model_config']
    for layer_name in f.attrs['layer_names']:
        qg = model_weights_group.create_group(layer_name)
        g = f[layer_name]
        qg.attrs['weight_names'] = g.attrs['weight_names']
        for weight_name in g.attrs['weight_names']:
            print(weight_name)
            weight_value = g[weight_name].value
            # plot_histograms(weight_name, weight_value)
            weight_value = weight_value.astype(np.float16)
            # plot_histograms(weight_name, weight_value)
            param_dest = qg.create_dataset(weight_name, weight_value.shape, dtype=np.float16)
            param_dest[:] = weight_value
    #         name = str(weight_name).split('/')[-1]
    #         if name.split(':')[0] == 'bias':
    #             # plot_histograms(weight_name, weight_value)
    #             # quantized = google_quantize(weight_value, bits=32)
    #             quantized = linear_quantize(weight_value, bits=32)
    #             # print(quantized)
    #             # plot_histograms(weight_name, quantized)
    #             param_dest = qg.create_dataset(weight_name, quantized.shape, dtype=np.int32)
    #             param_dest[:] = quantized
    #         else:
    #             # quantized = google_quantize(weight_value, bits=8)
    #             quantized = linear_quantize(weight_value, bits=8)
    #             # print(quantized)
    #             # plot_histograms(weight_name, quantized)
    #             param_dest = qg.create_dataset(weight_name, quantized.shape, dtype=np.int8)
    #             param_dest[:] = quantized
    if model.attrs['training_config']:
        qmodel.attrs['training_config'] = model.attrs['training_config']
    qmodel.flush()
