import json  
import urllib
import tensorflow as tf
from firebase import firebase
import time

firebase = firebase.FirebaseApplication('https://piklappdev.firebaseio.com', authentication=None)

def newEntry(image):
	jsonImage = json.dumps(image)
	Backwards = json.loads(jsonImage)
	for key, value in Backwards.items():
		URLFirebase = "https://firebasestorage.googleapis.com/v0/b/piklappdev.appspot.com/o/imagestmp%2F" + key + ".jpg?alt=media&token=" + value
		urllib.urlretrieve(URLFirebase, "image.jpg")
		image_data = tf.gfile.FastGFile('image.jpg', 'rb').read()
		label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]
		with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

			with tf.Session() as sess:
				softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
				predictions = sess.run(softmax_tensor, \
					{'DecodeJpeg/contents:0': image_data})

			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

			human_string = label_lines[top_k[0]]
			score = predictions[0][top_k[0]]
			scoreString = str(score)
			firebase.put('/results/' + key,human_string, scoreString)
			firebase.delete("images/", key)

while True:
	results = firebase.get('images', None, params={'print': 'pretty'}) 
	if results != None:
		newEntry(results)

'''
https://firebasestorage.googleapis.com/v0/b/piklappdev.appspot.com/o/imagestmp%2F748.jpg?alt=media&token=020b26cc-b8cd-46a9-b358-72b61b24be67
https://firebasestorage.googleapis.com/v0/b/piklappdev.appspot.com/o/imagestmp%2F627.jpg?alt=media&token=c2bcf2f8-6643-4ab3-9f4f-11f6fe8a6597
'''