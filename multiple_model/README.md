
## How to run:

- Train
	* Albert on news style transfer:
	  ```
	  python main.py --task='news_china_taiwan' --epochs=[EPOCH_NUM] --batch_size=[BATCH_SIZE] --training --use_albert --transformer_model_size 312 --latent_size 312
	  ```
	
	* Tiny bert on yelp:
		```
	  python main.py --task='yelp' --epochs=[EPOCH_NUM] --batch_size=[BATCH_SIZE] --training --use_tiny_bert --transformer_model_size 256 --latent_size 256
	  ```

	
	* Tiny bert on politicals(democratic to republican):
	 	```
	  python main.py --task='political' --epochs=[EPOCH_NUM] --batch_size=[BATCH_SIZE] --training --use_tiny_bert --transformer_model_size 256 --latent_size 256
	  ```
	* Distil bert on yelp:
		```
	  python main.py --task='yelp' --epochs=[EPOCH_NUM] --batch_size=[BATCH_SIZE] --training --use_distil_bert --transformer_model_size 768 --latent_size 768
	  ```

	
	* Distil bert on politicals(democratic to republican):
	 	```
	  python main.py --task='political' --epochs=[EPOCH_NUM] --batch_size=[BATCH_SIZE] --training --use_distil_bert --transformer_model_size 768 --latent_size 768
	  ```


- eval
	* You need to train the model first
	* ```
	  python main.py --task=[TASK TYPE] --batch_size=[BATCH_SIZE] --[eval_positive, eval_negative]
	  ```

Note: The codes are similiar to codes in 'Bert'
