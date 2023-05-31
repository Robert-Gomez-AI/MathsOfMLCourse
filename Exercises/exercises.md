John Robert Gomez
 Matemáticas para machine learning

# Ejercicios

## Exercise 1.2
Suppose that we use a perceptron to detect spam messages. Let's say that each email message is represented by the frequency of ocurrence of keywords, and the output is $+1$ if the message is considered spam.

1.  Can you think of some keywords that will end up with a large positive weight in the perceptron?
    b) H ow a bout keywords that wil l get a negative weight?
    c) What parameter in the perceptron d i rectly affects how many border­
    line messages end up being classified as spam ?

## Exercise 1.3

The weight update rule in **(1.3)** has the nice interpretation that it moves in the direction of classifying $\textbf{x}(t)$ correctly. 

1. Show that $y(t)\textbf{W}^{T}(t)\textbf{x}(t)<0$. $y(t)\textbf{W}^{T}(t)\textbf{x}(t)<0$
2. Show that $y(t)\textbf{W}^{T}(t+1)\textbf{x}(t)> y(t)\textbf{W}^{T}(t) \textbf{x}(t)$. 

3. As fat as classifying  $ \textbf{x}(t)$ is concerned,argue that the move from $\textbf{w}(t)$ to $\textbf{w}(t+1)$ is a move 'in the right direction'.