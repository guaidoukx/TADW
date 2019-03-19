# TADW

paper: Network Representation Learning with Rich Text Information

I have roughly complete the algorithm, but there're still some problems not matching with the paper. And I don't know how to solve them.

I have accomplish the core algorithm in two ways, manually derivate and using pytorch, which are figured out clearly by the function names.

I just run on cora data.

Problems:

1. The training process is not robust. Some times it converges from millions to hunderds in one iteration, while sometimes it coverges realiy slowly. By the way, if I replace ![](http://latex.codecogs.com/gif.latex?\\min_{W,H}\\left\\|M-W^{T}HT\\right\\|_{F}^{2}+\\frac{\\lambda}{2}\\left(\\|W\\|_{F}^{2}+\\|H\\|_{F}^{2}\\right)) with ![](http://latex.codecogs.com/gif.latex?\\min_{W,H}\\quad\\text{avg}(\\left\\|M-W^{T}HT\\right\\|_{F}^{2})+\\frac{\\lambda}{2}\\left(\\|W\\|_{F}^{2}+\\|H\\|_{F}^{2}\\right)), the process is under expectations.
2. The objective can't converges in 10 iterations(which is mentioned in the paper). After about 400 iterations, the objective is still decreasing. (All the parameters are same with those in the paper)
3. The classification result is not satisfying. What is strange is that, the more iterations it trains, the worse the classification result is. I think it maybe because I don't tune the hyper-paramters in SVM models.(But I don't know how.)

If you knoe how to cope with them, please contact me! Thanks a lot.



References:

[Another python code of this algorithm in GitHub](https://github.com/benedekrozemberczki/TADW). 

