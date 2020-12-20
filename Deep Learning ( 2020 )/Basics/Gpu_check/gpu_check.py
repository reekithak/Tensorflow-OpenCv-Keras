
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


print("Tf Version: ",tf.__version__)


print('Physical GPU : ',tf.config.list_physical_devices())
if(tf.test.is_gpu_available()):
    print("GPU Available ?: ",tf.test.is_gpu_available())
else:
    print("no device detected")


print("Cuda Avaialble ?: ",tf.test.is_built_with_cuda())

print("Device Name: ",tf.test.gpu_device_name())
