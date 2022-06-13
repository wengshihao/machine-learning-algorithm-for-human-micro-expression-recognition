# Machine-Learning-Algorithm-for-Human-Micro-Expression-Recognition
Code of My Undergraduate Thesis in [School of AI&CS, JNU](http://ai.jiangnan.edu.cn/) supervised by [Heng-Yang Lu](http://iip.nju.edu.cn/index.php/Luhy)
## 1. Four dilemmas in micro-expression recognition
**No.1** Face diversity dilemma, specifically refers to the human face of different individuals, facial differences and different posture of the problem.

**No.2** Behavioral spontaneity dilemma, specifically refers to the micro-expression of human uncontrolled spontaneous behavior, resulting in the amplitude
Small enough to be difficult to detect.

**No.3** Behavior transience dilemma, it refers to the problem of short duration of micro-expression (0.04s ~ 0.3s).

**No.4** "Human-machine collaboration" dilemma, specifically refers to the high time cost and low efficiency of manually rechecking each micro-expression recognized by the computer.

## 2. The solution to the face diversity dilemma
   Aiming at the problem of different poses, my paper uses affine transform to affine align the picture face and perform preliminary clipping, and get the face micro-expression video with correct pose and proper proportion. Aiming at the difference of individual face shape and facial features, my paper uses LWM face registration algorithm to adjust the pixels of local areas of key points of face, so that all faces in the dataset are as similar as possible to the selected model face in face shape and facial features position.
   
$\color{red}{LWM.py}$ is an implementation of the above text.

**You can run this code by setting the following parameters in the *main* function:**
```python
if __name__ == "__main__":
    #parameters
    videopath='EP18_01.avi'
    standardfacepath='standardface.jpg'
    savepath='reg.mp4'
```
## 3. The solution to the behavioral spontaneity dilemma
   For dilemma 2, that is, the spontaneity of micro-expressions leads to problems of small amplitude and difficult to detect. In my paper, *Euler Video Magnification algorithm* is used to filter micro-expression action signals on the Pyramid of Laplace and then magnify and synthesize the strategy to achieve the required action magnification effect.

$\color{red}{EVM.py}$ is an implementation of the above text.

**You can run this code by setting the following parameters in the *main* function:**
```python
if __name__ == "__main__":
    #parameters
    videopath='EP18_01.avi'
    savepath='mag.mp4'
```

**You can also learn *Euler Video Magnification algorithm* by [Visit This Page](https://nbviewer.org/github/yourwanghao/CMUComputationalPhotography/blob/master/class7/notebook7.ipynb) which is a Homework Assignment in **CMU**.**

## 4. The solution to the behavior transience dilemma
In view of dilemma 3, that is, the short duration caused by the brevity of micro-expressions, my paper uses *Time-domain Interpolation algorithm* to restore the high-dimensional curve of realistic micro-expressions through the existing video frames, and carries out more intensive resamplings on this curve to improve the frame number of micro-expressions.

$\color{red}{TIM.py}$ is an implementation of the above text.

**You can run this code by setting the following parameters in the *main* function:**
```python
if __name__ == "__main__":
    #parameters
    videopath='EP18_01.avi'
    savepath='tim.mp4'
```
**You should adjust the frame interval in the *process* function for the specific video：**
```python
res = tim.run(frames[5:51], 85)
```

**You can also learn *Time-domain Interpolation algorithm* by read [This Paper](https://readpaper.com/pdf-annotate/note?noteId=668667091142946816&pdfId=668666968103038976) which is from CVPR2011.**

## 5. Feature extraction
In my paper, LBP-TOP feature extraction algorithm is used for feature extraction of the microexpression video after the above processing.

$\color{red}{LBP\_TOP.py}$ is an implementation of the above text.

**You can run this code by setting the following parameters in the main function:**
```python
if __name__ == "__main__":
    #parameters
    videopath='EP18_01.avi'
    savepath='tim.mp4'
```


