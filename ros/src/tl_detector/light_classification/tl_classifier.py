from styx_msgs.msg import TrafficLight
import tensorflow as tflow
from PIL import Image
import numpy as np
import os
import rospy

class TLClassifier(object):
    def __init__(self):

	pass

 

    def run_inference_for_single_image(self, image, graph):
	  with graph.as_default():
	    with tflow.Session() as sess:
	      # Get handles to input and output tensors
	      ops = tflow.get_default_graph().get_operations()
	      all_tensor_names = {output.name for op in ops for output in op.outputs}
	      tensor_dict = {}
	      for key in [
		  'num_detections', 'detection_boxes', 'detection_scores',
		  'detection_classes', 'detection_masks'
	      ]:
		tensor_name = key + ':0'
		if tensor_name in all_tensor_names:
		  tensor_dict[key] = tflow.get_default_graph().get_tensor_by_name(
		      tensor_name)
	      if 'detection_masks' in tensor_dict:
		# The following processing is only for single image
		detection_boxes = tflow.squeeze(tensor_dict['detection_boxes'], [0])
		detection_masks = tflow.squeeze(tensor_dict['detection_masks'], [0])
		# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
		real_num_detection = tflow.cast(tensor_dict['num_detections'][0], tflow.int32)
		detection_boxes = tflow.slice(detection_boxes, [0, 0], [real_num_detection, -1])
		detection_masks = tflow.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
		    detection_masks, detection_boxes, image.shape[0], image.shape[1])
		detection_masks_reframed = tflow.cast(
		    tflow.greater(detection_masks_reframed, 0.5), tflow.uint8)
		# Follow the convention by adding back the batch dimension
		tensor_dict['detection_masks'] = tflow.expand_dims(
		    detection_masks_reframed, 0)
	      image_tensor = tflow.get_default_graph().get_tensor_by_name('image_tensor:0')

	      # Run inference
	      output_dict = sess.run(tensor_dict,
		                     feed_dict={image_tensor: np.expand_dims(image, 0)})

	      # all outputs are float32 numpy arrays, so convert types as appropriate
	      output_dict['num_detections'] = int(output_dict['num_detections'][0])
	      
	      output_dict['detection_classes'] = output_dict[
		  'detection_classes'][0].astype(np.uint8)
	      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	      
	      output_dict['detection_scores'] = output_dict['detection_scores'][0]
	      #print (output_dict['detection_scores'])
	      if 'detection_masks' in output_dict:
		output_dict['detection_masks'] = output_dict['detection_masks'][0]
		
	  return output_dict

    def load_image_into_numpy_array(self, image):
	  im_width = image.shape[0]
          im_height = image.shape[1]
	  return np.array(image.reshape(im_height, im_width, 3))


    def get_classification(self, image, graph):
        print "in classification module"
	output_dict = self.run_inference_for_single_image(image, graph)
	color_dict = {1: {'name': 'red', 'id': 1}, 2: {'name': 'yellow', 'id': 2}, 3: {'name': 'green', 'id': 3}}
	i=0
	ids = [i for i in range(len(output_dict['detection_scores'])) if output_dict['detection_scores'][i] > 0.5]
	print (ids)
	#print(output_dict['detection_scores'])
	#print(output_dict['detection_classes'])
	classes = [output_dict['detection_classes'][element] for element in ids]
	print (classes)

	#print (color_dict[1]['name'])

	output_colors = [color_dict[element]['name'] for element in classes]
	print (output_colors)
	
	if classes.count(1) > 0:
		if ((classes.count(2) >0) or (classes.count(3) >0)):
                   return TrafficLight.UNKNOWN 
                else:
                   return TrafficLight.RED
        if classes.count(2) > 0:
                if ((classes.count(1) >0) or (classes.count(3) >0)):
                   return TrafficLight.UNKNOWN 
                else:
                   return TrafficLight.YELLOW 
        if classes.count(3) > 0:
                if ((classes.count(1)>0) or (classes.count(2)>0)):
                   return TrafficLight.UNKNOWN  
                else :
                   return TrafficLight.GREEN
        if ((classes.count(1)==0) and (classes.count(2)==0) and (classes.count(3)==0)):
                return TrafficLight.UNKNOWN

