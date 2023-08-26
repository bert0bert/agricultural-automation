# Evaluating Edge & Cloud Computing for Automation in Agriculture

This experiment compares the latency between Edge and Cloud computing in an Agriculutral setting-- specifically in human detection.

It should take 2-3 hours to run this experiement.

You can run this experiemnt on the [Chameleon](https://chameleoncloud.org/) testbed. To run this experiment you should already have an account on Chameleon, be part of a project, and have configured keys om CHI@Edge, CHI@UC, and KVM@TACC.

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

![image_720](https://github.com/bert0bert/agricultural-automation/assets/141275632/13b3b507-bf81-48cf-b1b0-11f93fdd6b87)




## Run my experiment

### Set up resources at CHI@Edge
first least for raspi4
set up server
insteall ml

### Measure inference time at CHI@Edge
how to copy models to chi@edge provide test images etc

add folder, name it edge, in folder put basic tflite model edgetpu, text.txt, test images

### Set up resources at CHI@UC
first lease for rtx6000gpu
set up server
install machine learning

### Measure inference time at CHI@UC

how to copy models to chi@edge provide test images etc

add folder, name it edge, in folder put h5 model, text.txt, test images

### Set up resources at KVM@TACC
set up stuff

### Measure network transfer times at KVM@TACC

set it up how it is on my computer



## Notes

### References
copy refrences from poster

### Acknowledgements
i'd like to acknowledge support of NYU TANDON, K12 STEM outreach center, The Pinkterton Foundation, and my mentors at the New York State Center for Advanced Technology in Telecommunications: Chandra Shekhar Pandey and Fatih Berkay Sarpkaya
