# Evaluating Edge & Cloud Computing for Automation in Agriculture

This experiment compares the latency between Edge and Cloud computing in an Agriculutral setting-- specifically in human detection.

It should take 2-3 hours to run this experiement.

You can run this experiemnt on the [Chameleon](https://chameleoncloud.org/) testbed. To run this experiment you should already have an account on Chameleon, be part of a project, and have configured keys on CHI@Edge, CHI@UC, and KVM@TACC.

## Background

### Future of Agriculutral
New technology can help increase productivy/automation in agriculture. For example, new wireless network technology have the ability to produce ultra fast data transfer speeds within a private 5G network. Robotics and artificial intelligence have the ability to allow humans and robots to work together in an agricultural setting. All these things require computations, but where should these computations be placed? In the edge or in the cloud?


### Edge vs Cloud
In edge computing, input data (in this case an image) is sent to a machine learning (ML) model inside the edge device. The ML model then takes time to make a prediction, that time is called "Inference Time". From that prediction, an action is taken. In this case our action would be to continue moving forward or stop to avoid an accident. 

In cloud computing, our input data taken by the edge device is now sent over to a ML model inside a cloud server. The time it takes to send that input data over to our cloud server is called "Network Transfer Time". Our ML model then takes Inference Time to make a prediction and once the prediction is made it's then sent over to our edge device to take an action-- again taking Network Transfer Time. 

With these two possibilites come tradeoffs, while edge devices are slower when it comes to Inference Time because the device themselves are low-resource devices)-- there is almost little to no Network Transfer Time because data isn't being sent anywere. In cloud devices, Network Transfer Time may be slow due to pending network conditions, but Inference Time is usually faster because it's a more powerful device. With these possibilites, we have to test which is more efficient.


### Methodology
The [NREC Person Detection Dataset](https://www.nrec.ri.cmu.edu/solutions/agriculture/other-agriculture-projects/human-detection-and-tracking.html) best reflects what we're looking for in a dataset which is the ability to detect humans in an agricultural setting. With this dataset I trained a model on Google's [Teachable Machines](https://teachablemachine.withgoogle.com/) a simple way to train a model by Google. I created two classes, one called "human" and the other "no human" and uploaded the appropriate images to each set. With our ML model trained and saved to Google Drive, I then had to find scenarios for Network scenarios in which new 5G technology is in use. For this we chose mmWave link traces; I found data already collected from my lab and used this to create Network Transfer Times for my cloud scenarios. The inference devices selected for this experiment are as follows: a Raspberry Pi 4 with CPU capabillities, Google's Coral Dev Board (one scenario using CPU capabilites and the other using TPU capabilities) for our edge scenario; and A GPU (RTX6000) for our cloud scenario. Our cloud scenario has two examples: one with and without optimizations. These optimizations are meant to produce faster inference times. 

## Results

Below are the results from the experiment. I will go over each graph and explain what they mean. 

![image_720](https://github.com/bert0bert/agricultural-automation/assets/141275632/d8c4294a-d276-417d-8fcc-24f4c2ded521)

The purple is meant to show Inference Times and the blue network transfer times. On the left are the first three edge devices, we simply put our inference times into a google sheets, downloaded it and made a graph on Google colab. Left of these three devices are the four scenarios using GPU capabilites and GPU + Optimization. You can also see we have a scenario where there is and isn't blockage. This is simply us trying to recreate mmWave wireless links in real life as these links can be blocked easily. For this we got the inference time and added it to the two different Network Transfer Times which is done seperately on Google Colab. One network transfer time of 5ms (which is meant to represent no blockage in connection) and the other of 10ms (which is meant to represent a possible long period of blockage in the connection). The Raspberry Pi 4 and the Coral Dev Board with CPU capabilites were the slowest-- taking longer than 40ms, the Cloud GPU with and without blockage are third fastest, followed by Cloud GPU + Optimizations with and without blockage, and the Coral Dev Board with TPU capabilites being the fastest at less than 5 ms. 

![Screenshot 2023-08-25 at 9 46 25 PM](https://github.com/bert0bert/agricultural-automation/assets/141275632/bd7cba9e-fceb-4070-8111-8ddcdf993947)

This graph represents the network trasnfer times. Specifically in a setting where there are "long blockages". Since mmWave wireless links can be blocked, there must be scenarios where the connection must be blocked for a long period of time. This graph represents that example of there being instances of long blockages. The median transfer time was around 4-5ms and in cases where the link was blocked for a long period of tmie the transfer time would go up to 10ms. 

## Run my experiment

### Set up resources at CHI@Edge

First make sure you have a lease for a Raspberry Pi 4, you can find this if you go to the GUI for CHI@Edge, click on reservations, and it should be under leases. If not create a reservation for a Raspberry Pi 4. Once you have the lease, open this [link](https://github.com/teaching-on-testbeds/edge-cpu-inference/) to access the Edge inferencing on CPU. This will allow us to set up an experiment on Jupyter using a Raspberry Pi 4 to make inferences. 

Once you have a Jupyter Notebook set up you should follow the tutorial, it only takes around 10-15 minutes in total to complete. If you're having trouble finding the lease_id, go back to the Chameleon page where you found if your lease exists, click on lease itself (it should be highlighted in blue) and copy and paste the "Id". Note: "project Id" is something different, do not confuse it for the lease ID. Once you finish the tutorial we are now ready to begin testing images.

Before we go on to the next part of the tutorial, we first need to install our ML model. Click on the human_detection_tflite folder and download the contents inside to your computer as well as the test images called "positives_negatives". 

### Measure inference time at CHI@Edge

On Jupyter, go to the folder that says "image_model". In here, we're going to first delete the original model, labels, and image file by left-clicking on it and pressing delete. The model should be called "model.tflite", if it isn't look for any file that ends in ".tflite". Once we deleted those things we can either drag and drop or copy and paste our "human_detection.tflite" ML model, our labels.txt file, and the test images that we downloaded from this repository (there should be pos1-5 and neg1-5). Now that we have these things downloaded we need to change a few things to begin testing. Go to model.py and first change line 13. Where it says "model_path=" in the single quotes write 'human_detection.tflite'. It should look something like this: '''interpreter = tflite.Interpreter(model_path='model.tflite')'''. Next, scroll to the bottom where it says "labels[i].split". Make sure inside the parenthesis next to split is empty. Don't delete it, just leave it empty. Finally on line 20 where it says "image_path = 'image_name'". We're going to rename 'image_name' everytime we're doing a different image. (image_path = 'pos_5.png') is an example of what it'll look like. 'pos4.png' would be another possibility. Now that we have these things we're now going to test different images and get different times. 

Your Jupyter environment should be set up, if it isn't make sure it's all set up. Whenever we test a new image, we begin runing code from the "Transfering files to the container" section. This is to make sure we're sending the right images to our ML model. Before we print our results, in the code above be sure to change the '''image_model/'image_name' ''' to the corresponding image number that you put in the model.py. Now we're ready to see what our model says it predicted and the time it takes to make this prediction. Write this time down with the appropriate images.

Re-do these steps for images 1-10. 

### Set up resources at CHI@UC

First make sure you have a lease for an RTX6000 GPU, you can find this if you go to the GUI for CHI@UC, click on reservations, and it should be under leases. If not create a reservation for a RTX6000 GPU. Once you have the lease, open this [link](https://github.com/teaching-on-testbeds/cloud-gpu-inference) to access the Cloud inferencing on GPU. This will allow us to set up an experiment on Jupyter using an RTX6000 to make inferences. 

Once you have a Jupyter Notebook set up you should follow the tutorial, it only takes around 10-15 minutes in total to complete. If you're having trouble finding the lease_id, go back to the Chameleon page where you found if your lease exists, click on lease itself (it should be highlighted in blue) and copy and paste the "Id". Note: "project Id" is something different, do not confuse it for the lease ID. Once you finish the tutorial we are now ready to begin testing images.

Before we go on to the next part of the tutorial, we first need to install our ML model. Click on the human_detection_keras folder and download the contents inside to your computer. You should also still have the test images downloaded to your computer. 


### Measure inference time at CHI@UC

On Jupyter, go to the folder that says "image_model". In here, we're going to first delete the original model, labels, and image file by left-clicking on it and pressing delete. The model should be called "model.h5", if it isn't look for any file that ends in ".h5". Once we deleted those things we can either drag and drop or copy and paste our "keras_model.h5" ML model, our labels.txt file, and the test images that we downloaded from this repository (there should be pos1-5 and neg1-5). Now that we have these things downloaded we need to change a few things to begin testing. 

In model.py:
- Change model = tf.keras.applications.MobileNetV2(input_shape=INPUT_IMG_SHAPE) to model = tf.keras.saving.load_model('model.h5')
- Change image_path = 'parrot.jpg' to replace with whatever the name of your test image is.
- Change imagenet_labels = np.array(open(url).read().splitlines())[1:] to imagenet_labels = np.array(open('labels.txt').read().splitlines())[1:]

In model-convert.py:
- Change model = tf.keras.applications.MobileNetV2(input_shape=INPUT_IMG_SHAPE) to model = tf.keras.saving.load_model('model.h5')

In model-opt.py:
- Change image_path = 'parrot.jpg' to replace with whatever the name of your test image is.
- Change imagenet_labels = np.array(open(url).read().splitlines())[1:] to imagenet_labels = np.array(open('labels.txt').read().splitlines())[1:]
- you may also have to change the word: predictions in top_3 = np.argsort(output["predictions"].numpy().squeeze())[-3:][::-1] and print('{:.6f}'.format(output["predictions"].numpy()[0, i]), ':', imagenet_labels[i]) to sequential_3

Your Jupyter environment should be set up, if it isn't make sure it's all set up. Whenever we test a new image, we begin runing code from the "Transfering files to the container" section. This is to make sure we're sending the right images to our ML model. Before we print our results, in the code above be sure to change the '''image_model/'image_name' ''' to the corresponding image number that you put in the model.py. Now we're ready to see what our model says it predicted and the time it takes to make this prediction. Write this time down with the appropriate images.

Re-do these steps for images 1-10. 

### Set up resources at KVM@TACC

First you must make sure you're able to set up instances at KVM@TACC. If you have access to this we can continue.

Once you're certain you have access to KVM@TACC, open this [link](https://github.com/teaching-on-testbeds/network-emulation) network emulation. This will allow us to set up an experiment on Jupyter and runs us through a quick tutorial on how to use network emulation. 

The tutorial takes around 10-15 minutes to run through, please complete the tutorial before continuing to set up the mmWave Wireless link trace. 


### Measure network transfer times at KVM@TACC

Now that you understand how to use KVM@TACC, we're going to set up our network to emulate a mmWave wireless link trace. 
If you'd like to take a look on the data my lab collected, please follow this [link](https://witestlab.poly.edu/blog/tcp-mmwave/) which provides the steps necessary to recreate a mmWave scenario. Only follow the "Set Up Resources" section, the rest will be done here. 

*When configuring your settings in router, make sure you change the IP address to the correct one in the tputvary.sh bash script.

Now we have to upload our images to Romeo so they can be sent to Juliet.
Run this code on your Jupyter notebook:
- remote_romeo.run("sudo apt update; sudo apt install apache2")

*In a terminal on your LAPTOP - transfer files to home directory on romeo -

- scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa_chameleon -r  /Users/albertonajera/Desktop/positives cc@129.114.27.1:~/

Once the images are in Romeo, we are now going to begin sending images from Romeo to Juliet through our mmWave configured router. On router we're going to run one of four commands:
- bash tputvary.sh sl  
- bash tputvary.sh sb  
- bash tputvary.sh lb  
- bash tputvary.sh mobb

What these commands do will emmulate a specific network scenario in mmWireless link. sl being static link, sb = short blockage, ls = long blockage, and mobb = mobility. When we run one of these we'll quickly go to juliet and run one of four commands:
 
- curl -so /dev/null -w "%[time_total}\n" http://10.10.1.100/positives_negatives/post.1png?[1-2000000000] &> /dev/stdout sl-results.txt 
- curl -so /dev/null -w "%[time_total}\n" http://10.10.1.100/positives_negatives/post.1png?[1-2000000000] &> /dev/stdout sb-results.txt
- curl -so /dev/null -w "%[time_total}\n" http://10.10.1.100/positives_negatives/post.1png?[1-2000000000] &> /dev/stdout lb-results.txt
- curl -so /dev/null -w "%[time_total}\n" http://10.10.1.100/positives_negatives/post.1png?[1-2000000000] &> /dev/stdout mobb-results.txt

These commands will Send an image (specifically post1.png) back and forth from Romeo to Juliet and time the amount it takes each time. You will run each one with the corresponding mmWave scenario. sl will go with sl, sb with sb, etc. After running the command on Juliet let it run for 110 seconds. Once you finish running the command the txt files will download to your Jupyter environment. From there you can then download the files to your computer by left clicking on the txt file in your Jupyter notebook and downloading it.


## Notes

### References
Chameleon, NYU Tandon School of Engineering, Center for Advanced Technology in Telecommunications, the Pinkerton Foundation, Center for k12 STEM Outreach Program

### Acknowledgements
i'd like to acknowledge support of NYU TANDON, K12 STEM outreach center, The Pinkterton Foundation, and my mentors at the New York State Center for Advanced Technology in Telecommunications: Chandra Shekhar Pandey and Fatih Berkay Sarpkaya
