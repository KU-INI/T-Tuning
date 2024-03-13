# T-Tuning

## Architecture of T-Tuning
<p align="center"><img src="https://github.com/KU-INI/T-Tuning/assets/109642935/388bcf0e-9884-4a50-8cff-3e6db0ba05dd" width="450" height = "300"/></p>


## Experiment result
### GLUE
# |Method|Trainable Parameter|	CoLA|	SST-2|	MRPC|	STS-B|	QQP|	MNLI|	QNLI|	RTE|	Avg.||
# |------|		---|			---|	---|	---|	---|	---|	---|	---|	---|	---|
# |Full Fine-tuning|	355M|			68|	96.4|	90.9|	92.4|	92.2|	90.2|	94.7|	86.6|	88.92|
# |LoRA|		0.8M|            		68.2|	96.2|	90.9|	92.6|	91.6|	90.6|	94.8|	87.4|	89.03|
# |T-Tuning(rank : 1)|	0.14M|			96.5|	96.7|	92.2|	91.9|	89.7|	90|	94.3|	88.4|	89.08|
