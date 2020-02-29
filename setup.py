import tensorflow as tf
import tensorflow_hub as hub

url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(url).signatures['default']

tf.saved_model.save(detector, "/tmp/faster_rcnn/1/")
