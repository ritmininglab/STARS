######################################################################################################################
STARS.py includes all the customized functions and dependencies needed to conduct the active learning with re-sampling.

Data format:
Input data X should be a (float)N * M numpy array(Design matrix). 
Label y should be a (int)N*1 numpy array. The K classes should be coded ranging from 0 to K-1.

Function description:
	1.partitionForAL() -> trainIndex,testIndex,poolIndex
	This function takes entire label vector and split the dataset in three parts: training, pool, and testing.
		leastTrain: the number of data instances per class preserved for training.
		leastPool: the number of data instances per class preserved for candidatePool.
	
	2.simpleALsvm()-> res,trainIndex,testIndex,poolIndex,resampleIter,resampleIndex:
	This function runs active learning with re-sampling.
		clf: The classifier model. clf should be a instance of scikit-learn estimator. 
			 To pass a customized classifier, one should make sure clf passes the is_classifier test: sklearn.base.is_classifier(clf)=True
			 To use pre-defined classifier such as support vector machines, logistic regression one can create the estimator according to official API:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass
		noiseLv: The noise level, alpha, of the annotator. 
		y0: The clean labels. This is only introduced for experimental purpose.(e.g. compute the noise detection coverage, etc.)
			In reality, since we don't know the true labels, it is ok to set y0=y
		sample: The number of active learning iterations. The value is bounded by the size of unlabeled pool. i.e. sample<=len(poolIndex)
		method: The sampling method taken by active learning. Since active sampling is not the major study object of this work, we only provide 'bvs' as best-vs-second best.
		resampleFreq: Controls the re-sample frequency. The total annotation budget is sample + int(sample/resampleFreq).
		resampleMethod: The re-sampling methods. Options are 
			'DEC':Decision function based re-sampling.
			'LOSS':Loss function based re-sampling.(Note the model loss is hinge for SVMs. Other types of loss need to be defined in advance)
			'STARS':The proposed re-sampling stratege.
			'Random':Random re-sampling
		resampleCap: The maximum number of re-labeling a certain data point. If the selected sample has meet the resampleCap, the secondary data point in the re-sample list will be re-labeled.
		EMAcoef:The coefficient(gamma) that controls the exponential moving avarage in STATS.
		EMAscope:The threshold that controls how many training instances will receive attentions from the model and keep the previous STARTS scores.
		tau1,tau0: Parameters that control the interval of DEC-LIC balancing parameter, tau. The value of tau would linearly change from tau1 to tau0 during the AL process.
		
		