import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2
import rospy
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sensor_msgs

bridge = CvBridge()

def img_callback(msg):
    try:
        # cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        cv_image = imgmsg_to_cv2(msg,dtype=np.uint8)
        print(cv_image.shape)
        cv_image = cv_image[:,:,:3]
        # cv_image = cv2.resize(cv_image, (960, 540)) 
        # cv2.imwrite('ros_test.jpg', cv_image[:,:,0])
    except CvBridgeError as e:
        print(e)
        return

    depth = depth_anything.infer_image(cv_image, args.input_size)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)
    depth = depth.astype(np.float32)

    
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

    cv2.imwrite(os.path.join(args.outdir, 'frame_depth.jpg'), depth)
    depth_msg = cv2_to_imgmsg(depth, encoding="passthrough", header=None)

    depth_msg.header.frame_id = msg.header.frame_id
    depth_msg.header.stamp = rospy.Time.now()
    pub.publish(depth_msg)


def cv2_to_imgmsg(cvim, encoding="passthrough", header=None):
    if not isinstance(cvim, (np.ndarray, np.generic)):
        raise TypeError('input type is not a numpy array')
    
    img_msg = sensor_msgs.msg.Image()
    img_msg.height = cvim.shape[0]
    img_msg.width = cvim.shape[1]
    if header is not None:
        img_msg.header = header
    numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                            'int16': '16S', 'int32': '32S', 'float32': '32F',
                            'float64': '64F'}
    numpy_type_to_cvtype.update(dict((v, k) for (k, v) in numpy_type_to_cvtype.items()))
    if len(cvim.shape) < 3:
        cv_type = '{}C{}'.format(numpy_type_to_cvtype[cvim.dtype.name], 1)
    else:
        cv_type = '{}C{}'.format(numpy_type_to_cvtype[cvim.dtype.name], cvim.shape[2])
    if encoding == "passthrough":
        img_msg.encoding = cv_type
    else:
        img_msg.encoding = encoding
    if cvim.dtype.byteorder == '>':
        img_msg.is_bigendian = True

    img_msg.data = cvim.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    # print(img_msg)
    return img_msg


def imgmsg_to_cv2(img_msg, dtype=np.uint8):
    return np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, -1)


def main():
    global model, pub, depth_anything
    rospy.init_node('image_segmentation_node')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    rospy.Subscriber("/realm/alexa/img", Image, img_callback)

    # rospy.Subscriber("/realm/alexa/mosaicing/rgb", Image, img_callback)
    # rospy.Subscriber("/realm/alexa/ortho_rectification/rectified", Image, img_callback)
    pub = rospy.Publisher('/Anydepth_map/', Image, queue_size=1)
    
    rospy.spin()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    main()
