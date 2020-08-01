## Part 1:  Part-of-speech tagging
### Problem:
Find the parts of speech tags for a new sentence given you have a labelled data with pos tags already.

Given Parts of Speech (12) :
```python
['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
```

Different Models Used:

### 1. Simple

##### Training 

Find the count of all pos tags occurring for each word and find their Probabilities 

( This is used for finding the pos tags using simple model)

To perform part-of-speech tagging, we want to estimate the most-probable tags for each word Wi,

$$
s∗i= arg maxsi P(Si=si|W).
$$

Ex:

| Words   | 'adv' | 'adj' | 'verb' | .................. |
| ------- | ----- | ----- | ------ | ------------------ |
| The     | 10    | 2     | 3      | ................   |
| Prudhvi | 2     | 4     | 6      | ........           |
| Anji    | 3     | 5     | 4      | .........          |
| Ruthvik | 6     | 6     | 2      | .............      |

Similarly find their Probability table by Dividing wrt each column.

#### Testing

Take the maximum occurred pos tag for each word in the given sentence. 
If the is not their in training data set label it as noun (As nouns are the most common occurences for a new data point as all other pos tags are general occuring tags) and accuracy is also good :-)

### 2.HMM Viterbi

##### Training

Calculate the Emission and Transition  Probability tables

Emission Table:

Probability of Occurrence of a word given pos tag  P(tag = 't1'/word)

 if their zero probability give it a probability of 0.000000000001

| Words        | tag1           | tag2           | tag3 | ........ |
| ------------ | -------------- | -------------- | ---- | -------- |
| w1           | p(tag = t1/w1) | p(tag = t2/w1) | ...  | .......  |
| w2           | ..             | ..             | ..   | ..       |
| w3           | ....           | .              | .    | .        |
| ............ | .              | .              | .    | .        |

Transition Probability:

Probability of P(pos_tag2/pos_tag1)

Ex: P('noun'/'noun')

#### Testing
find the maximum a posteriori (MAP) labeling for the sentence

$$
(s∗1, . . . , s∗N) = arg   maxs1,...,sNP(Si=si|W).
$$

We can solve this model by using viterbi (Dynamic Programming) using transition and emission probabilities to calculate the maximun occuringsequence 


### 3.Complex_MCMC (Gibbs Sampling)

##### Training
Use the probability tables from previous viterbi and separetley caluculate the probaility of P(Sn/Sn-1,s0)


#### Testing

Initialize the word pos sequence to some random pos tags # Here I Initialized it to nouns.
Using Gibbs sampling sample the Probabilities each word by making all other values constant and after the healing period store the maximum occured sequences counts in a dictionary.

After sampling output the maximum occurred sequence for each word.

Testing Accuracies:

</b>Note: Accuracies for complex model may vary during runs as it takes random samples.</b>

For bc.test

| Models        | Words correct: | Sentences correct: |
| ------------- | -------------- | ------------------ |
| Ground truth: | 100.00%        | 100.00%            |
| Simple:       | 93.95%         | 47.50%             |
| HMM:          | 95.09%         | 54.40%             |
| Complex:      | 95.05%         | 54.30%             |

For bc.test.tiny

| Models        | Words correct: | Sentences correct: |
| ------------- | -------------- | ------------------ |
| Ground truth: | 100.00%        | 100.00%            |
| Simple:       | 97.62%         | 66.67%             |
| HMM:          | 97.62%         | 66.67%             |
| Complex:      | 100.00%        | 100.00%            |


#### Posterior Probabilites:

### Simple:
   Calculate the Posterior Probabilities of simple model as P = p(word/tag)*p(tag)
   
### HMM:
   For HMM model P = p(word/tag)*prob(tag/prev_tag)
   
### Complex_mcmc
   Calculate the Posterior Probabilities of complex model as P = p(word/tag)*p(tag/prev_tag)*p(next_tag/tag)
