import os
import cv2
from playsound import playsound
from gtts import gTTS
from time import sleep
# Generate a random name for the audio file
import random
def name_generator():
    ran = random.randint(1,5000)
    ran = str(ran)
    return ran
# Responsible for text to speech translation (in english)
def speak(text):
    tts = gTTS(text=text, lang="en")
    # save audio file
    new_name = name_generator()
    new_name= new_name+".mp3"
    tts.save(new_name)
    # play the audio file
    print("saving...............................")
    playsound(new_name)
    print("saying................................")
    try:
        os.remove(new_name) 
    except:
        print("i cant")

# Welcoming  message

speak("Hello everyone!")
speak("I am BOTA! a voice assistant robot for Mr Aalaa Aayedi")
speak("This is a fingerprint matching project with python")
speak("For our fingerprint sample we chose the right index finger of a male")
speak("Once a match is found, results will be displayed!")
speak("I'm off now! have a nice day!")

#sample
sample = cv2.imread("archive/SOCOFing/Altered/Altered-Hard/1__M_Right_index_finger_CR.BMP")

# Initializing
best_score = 0
filename = None
image = None

# Keypoints of the original image and the sample image, as well as matching points initialized at none
kp1, kp2, mp = None, None, None
# Launch 
print("-----WELCOME-----")
print("Launching program...")
sleep(2)
print("loading...")
sleep(5)
print("loading...")
sleep(5)
print("loading...")
sleep(5)
print("loading...")
sleep(5)
print("loading...")
# Looping throught the images under the folder 'Real'
for file in [file for file in os.listdir("archive/SOCOFing/Real")]:
    fingerprint_image = cv2.imread("archive/SOCOFing/Real/"+file)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image,None)

    #uses KNN 
    matches = cv2.FlannBasedMatcher({'algorithm':1, 'trees':10},{}).knnMatch(descriptors_1, descriptors_2, k=2)

    #find relative match
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
    
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

# Displaying Results
print("-----RESULTS-----")
print(f"Best Match: {filename}")
print("Score: " + str(best_score))
print("")
# Drawing the matches between sample and real image
print("-----MATCHING POINTS-----")
inp = input("Do you want to see the output? (y/n)")
if inp == 'y':
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("result: ", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("")
    print("-----SAVE INFO-----")
    s = input("Do you want to save the output? (y/n)")
    if s == 'y':
        with open("match.png", "a+") as f:
            f.write(f'Match found : {result}')
            f.write("\n")
            f.close()
    elif s == 'n':
        print("-----GOODBYE-----")
elif inp == 'n':
        print("-----GOODBYE-----")