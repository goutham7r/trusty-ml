Latest Work:
	
	In folder RL: main notebook to run is RL.ipynb


1. Same actions taken from all states
2. Structure of Reward Function
	* Sparsity
	* Terminal Reward
	* Additivity
	* Make it continuous, introduce wall cost
3. Incorporate softmax probability into reward/actions
	* p_t2 - p_t1
	* p_t2 - max(p_i2)_i!=t
4. Increase action space
5. Introduce priority sampling in replay buffer
6. Other problems:
	* Categorical actions