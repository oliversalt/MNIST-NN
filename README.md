# MNIST-NN
These are two files that I wrote using primarily Numpy. It is a neural network that identifies shapes from the MNIST database

I wrote lots of this coding using GPT-4 and I used a very similar format to Samson Zhang.Here is the youtube video that inspired this project https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1699s

The Training file was used to create the network and train it on the MNIST dataset. 
Here the best accuracy I could achieve was around 0.93 with 0.88 accuracy on the testing set. So not very overfit and fairly accurate. 

The display predict drawing file is used to then test network parameters on new data it hasn't seen. then also display it using matplotlib.
it mostly guesses these correctly if you run the file repetitively 

The Interactive testing file is where you can draw your own shapes then test how good the network is at categorising them correctly.
Here the network isn't as good at recognising the newly written digits. This may have something to do about the way they are generated.
The digits written by the user are slightly thinner than the ones in the MNIST database. I did my best to recreate them. 

After drawing 10 of each number (except number 9, which it doesn't recognise for some reason) I managed to get the accuracy to around 63%
here are the outputs it gave for each number I drew: 
Number, outputs, total correct outputs 
1:2152515111 5
2:2222222252 9
3:3333323333 9
4:4444449448 8
5:5555555365 8
6:9466266657 5
7:2277277777 7
8:3828682658 4
9:7888833473 0
0:0020002000 8
63% accuracy

This is my first machine learning project and I am quite happy with the outcome!
