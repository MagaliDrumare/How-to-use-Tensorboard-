
***README  Tensorboard*** 

>***Mnist dataset*** 
-http://yann.lecun.com/exdb/mnist/
>***Multilayer perceptron*** 
-http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
>***Convolutional Neural Network*** 
-http://cs231n.github.io/convolutional-networks/
>***Gradient Descent*** 
-https://www.coursera.org/learn/neural-networks-deep-learning/lecture/A0tBd/gradient-descent
>***Convolutional Neural Network*** 
-http://cs231n.github.io/convolutional-networks/
>***Tensorflow Introduction*** 
-http://web.stanford.edu/class/cs20si/lectures/slides_01.pdf
>***Tensorboard*** 
-https://www.tensorflow.org/get_started/graph_viz
>***Hands-On Tensorboard (Tensorflow dev summit 2017)***
-https://www.youtube.com/watch?v=eBbEDRsCmv4

***1-Création des logs :***
>-Nommer les éléments name="x"
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y")

>-Representation des variables dans Tensorboard
tf.summary.histogram("W1",W_conv1)
tf.summary.histogram("b1", b_conv1)
tf.summary.histogram("W2",W_conv2)
tf.summary.histogram("b2", b_conv2)

>-Représentation des layers dans Tensorboard 
with tf.name_scope("convolutional1"): 
with tf.name_scope("fc1"):
with tf.name_scope("dropout"):
with tf.name_scope("output"):

>-Représentation des indicateurs de performances 
with tf.name_scope("entropy"):   
tf.summary.scalar("entropy", cross_entropy)
with tf.name_scope ("accuracy"):

***2-Fusion des logs tensorboard***
>summ = tf.summary.merge_all()

>Initialiser l’ecriture du tensorboard 
#Initialize les variables 
        sess.run(tf.global_variables_initializer())
        writer=tf.summary.FileWriter("/tmp/tensorboard/2")
        writer.add_graph(sess.graph)

***3-Incrémenter les logs dans la phase de training*** 
-Exemple 1 :  
if i%100 == 0:
        s= summ.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(s,i)
-Exemple 2: 
For I in range (2001): 
batch=mnist.train.next_batch(100)
If I%5 ==0: 
s=sess.run(merges_summary, feed_dict={x: batch [0], y: batch[1]}
writter.add_summary(s,i)
sess.run(train_step, eed_dict={x: batch [0], y: batch[1]})


***4-Taper dans la console tensorboard --logdir /tmp/tensorboard/2***

***5-Aller à l’adresse : http://0.0.0.0:6006 dans le browser*** 

-Possibilité de suivi des logs 
--tf.summary.scalar 
--tf.summary.image 
--tf.summary.audio 
--tf.summary.histogram 
--tf.summary.tensor 

-Possibilité de suivre plusieurs modèles à la fois 
--Taper dans la console : tensorboard --logdir /tmp/mnist_tutorial

-Autre fonctionnalité de tensorboard Embeded Visualizer 
 
