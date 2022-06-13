# Machine-Learning-Algorithm-for-Human-Micro-Expression-Recognition
Code of My Undergraduate Thesis in [School of AI&CS, JNU](http://ai.jiangnan.edu.cn/) supervised by [Heng-Yang Lu](http://iip.nju.edu.cn/index.php/Luhy)
## 1. Four dilemmas in micro-expression recognition
**No.1** Face diversity dilemma, specifically refers to the human face of different individuals, facial differences and different posture of the problem.

**No.2** Behavioral spontaneity dilemma, specifically refers to the micro-expression of human uncontrolled spontaneous behavior, resulting in the amplitude
Small enough to be difficult to detect.

**No.3** Transient dilemma, it refers to the problem of short duration of micro-expression (0.04s ~ 0.3s).

**No.4** "Human-machine collaboration" dilemma, specifically refers to the high time cost and low efficiency of manually rechecking each micro-expression recognized by the computer.

## 2. The solution to the face diversity dilemma
   Aiming at the problem of different poses, this paper uses affine transform to affine align the picture face and perform preliminary clipping, and get the face micro-expression video with correct pose and proper proportion. Aiming at the difference of individual face shape and facial features, this paper uses LWM face registration algorithm to adjust the pixels of local areas of key points of face, so that all faces in the dataset are as similar as possible to the selected model face in face shape and facial features position.
   
$\color{red}{LWM.py is an implementation of the above text}$
<font color=#FF0000 > **LWM.py is an implementation of the above text** </font>
