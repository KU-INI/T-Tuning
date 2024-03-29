# T-Tuning

## Architecture of T-Tuning
<p align="center"><img src="https://github.com/KU-INI/T-Tuning/assets/109642935/388bcf0e-9884-4a50-8cff-3e6db0ba05dd" width="450" height = "300"/></p>


## Experiment result
### GLUE
We conducted experiments on the GLUE benchmark using RoBERTa-large.
|Method|Trainable Parameter|	CoLA|	SST-2|	MRPC|	STS-B|	QQP|	MNLI|	QNLI|	RTE|	Avg.|
|------|		---|			---|	---|	---|	---|	---|	---|	---|	---|	---|
|Full Fine-tuning|	355M|			68|	96.4|	90.9|	92.4|	92.2|	90.2|	94.7|	86.6|	88.92|
|LoRA|		0.8M|            		68.2|	96.2|	90.9|	92.6|	91.6|	90.6|	94.8|	87.4|	89.03|
|T-Tuning(rank:1)|	0.14M|			69.5|	96.7|	92.2|	91.9|	89.7|	90|	94.3|	88.4|	89.08|

### E2E NLG Challenge
We conducted experiments on the E2E NLG Challenge using GPT2-medium.
|Method|Trainable Parameter|	BLEU|	NIST|	METEOR|	ROUGE-L|	CIDEr|
|------|		---|			        ---|	---|	---|	  ---|	    ---|	
|Full Fine-tuning|	354.92M|	68.2|	  8.62|	46.2|	71.0|	2.47|
|LoRA|		0.35M|            	70.4|	8.85|	46.8|	71.8|	2.53|
|T-Tuning(rank:3)|	0.29M|	-|	-|	-|	-|	-|
