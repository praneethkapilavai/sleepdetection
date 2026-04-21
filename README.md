# 💡 Driver's Drowsiness Detection using OpenCV-Python

🤔 Ever fallen asleep while driving alone? 

🤔 Ever saved yourself from any accident by waking up instantly and looking in the side mirror and thanking god, because you hadn't completed your sleep?

🤔 Did you ever think to have someone who will give you an alert while you are feeling drowsy?

😉 Python comes for the rescue here!

👉 I made a python script that will give an alert to the driver when the driver is feeling drowsy


## Demo Output Video



## 💡 Implementation of the Logic

👉 Basically, I used the famous dlib library which can estimate the location of 68 coordinates (x, y) that map the facial points on a person’s face in real-time.<br>
You can download the library from <a href="https://github.com/Nisarg1112/Driver-s-Drowsiness-Detection-using-OpenCV-Python/tree/main/shape_predictor_68_face_landmarks.dat">here.</a>

**📸 Here is a Image to show 68 points on our face**

<img src="assets/68_landmark.jpeg" height=300px><br>

👉 Then, I am calculating Eye Aspect Ratio (EAR) of Left Eye and Right Eye using Euclidean Distance between 3 pairs of different coordinates of both the eyes as seen in 68 Points landmark's image!

👉 Then, I set a threshold value, If EAR goes below that threshold point, It will alert the driver!

👉 Planning to integrate Lips Detection Algorithm also, to detect the driver is yawning or not!

👉 I am thinking to deploy it on a raspberry pi that will have an alarm attached to it so it can be really helpful to someone

## ⚡️ How to Use

Just follow 4 simple steps:

1. Clone repository to preserve directory structure<br>
`git clone https://github.com/Nisarg1112/Driver-s-Drowsiness-Detection-using-OpenCV-Python.git`
2. Go to your favorite code editor and open Command Prompt (cmd) amd go to directory where you cloned this repo
3. Run this command in cmd<br>
   `pip install -r requirements.txt`
4. Run `face_eye_detection_for_drowsiness.py` and Enjoy😎!
   
**Note**: Incase, if you want to just detect the 68 points on Human face then run `face_landmark.py` file and You are good to go!😎

## 🙋‍♂️ Helpdesk

**If you face any problem like script not running in local environment or anything:** You can reach out to me at anytime on following platforms!
<br>
<br>

## ℹ References

The ideas presented in this repo came primarily from the Dlib library. The implementation here also took significant inspiration. The pretrained model used in this project came from the official Dlib website.

<ul type='square'>
  <li><strong>Davis.King - </strong><a href='https://github.com/davisking/dlib'>dlib C++ library</a></li>
  <li><strong>Italo José - </strong><a href='https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672'>Facial mapping (landmarks) with Dlib + python</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'><strong>The official dlib </strong><a href='http://dlib.net/'>website</a></a></li>
</ul>


