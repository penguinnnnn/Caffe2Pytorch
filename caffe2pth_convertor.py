"""Convert a Caffe model to Pytorch

Given a prototxt and a caffemodel, this code outputs a pth model.
You can reconstruct the network by the prototxt and the pth model.

Supported Caffe layers:
    'Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 'PReLU',
    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss',
    'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput'
    
***Notice: This code requires python2***
"""

from caffe2pth.caffenet import *
import argparse
import caffe.proto.caffe_pb2 as caffe_pb2


# Default input prototxt path
_PROTOTXT_PATH = '/mnt/lustre/share/shenyujun/models/106points/106points.prototxt'

# Default input caffemodel path
_CAFFEMODEL_PATH = '/mnt/lustre/share/shenyujun/models/106points/106points.caffemodel'

# Default output pytorch weights path
_PTHMODEL_PATH = '/mnt/lustre/share/shenyujun/models/106points/106points.pth'


def parse_caffemodel(caffemodel):
    """Parse a caffemodel
    
    Inputs:
      string of caffemodel path
    Returns:
      Parsed model
    
    This function should be in prototxt.py, but this function
    requires caffe which requires python2. So move it here.
    """
    model = caffe_pb2.NetParameter()
    print('Loading caffemodel: ' + caffemodel)
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model


def main():
    parser = argparse.ArgumentParser(
                       description='Convert caffe to pytorch.')
    parser.add_argument('--prototxt', type=str, default=_PROTOTXT_PATH,
                       help='Caffe prototxt path.')
    parser.add_argument('--caffemodel', type=str, default=_CAFFEMODEL_PATH,
                       help='Caffe caffemodel path.')
    parser.add_argument('--pthmodel', type=str, default=_PTHMODEL_PATH,
                       help='Output pytorch model path.')
    args = parser.parse_args()
    
    # Load network model
    print('========================================')
    print('Parsing Caffe model.')
    print('----------------------------------------')
    net = CaffeNet(args.prototxt)
    
    print('----------------------------------------')
    print('Print converted pytorch model.')
    print('----------------------------------------')
    print(net)
    
    # Load network weights
    print('----------------------------------------')
    print('Loading weights.')
    print('----------------------------------------')
    net.load_weights(parse_caffemodel(args.caffemodel))
    
    # Save model structure as OrderedDict
    print('----------------------------------------')
    print('Saving pytorch weights.')
    print('----------------------------------------')
    torch.save(net.state_dict(), args.pthmodel)
    
    print('End.')
    print('========================================')

    
if __name__ == '__main__':
    main()

