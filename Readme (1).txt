1. Bag of Words and MCP
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 50)                1900400   
                                                                 
 dense_1 (Dense)             (None, 1)                 51        
                                                                 
=================================================================
Total params: 1,900,451
Trainable params: 1,900,451
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
282/282 - 9s - loss: 0.3722 - accuracy: 0.8466 - 9s/epoch - 31ms/step
Epoch 2/10
282/282 - 6s - loss: 0.0974 - accuracy: 0.9728 - 6s/epoch - 20ms/step
Epoch 3/10
282/282 - 6s - loss: 0.0311 - accuracy: 0.9956 - 6s/epoch - 20ms/step
Epoch 4/10
282/282 - 6s - loss: 0.0108 - accuracy: 0.9991 - 6s/epoch - 20ms/step
Epoch 5/10
282/282 - 6s - loss: 0.0050 - accuracy: 0.9999 - 6s/epoch - 21ms/step
Epoch 6/10
282/282 - 6s - loss: 0.0029 - accuracy: 1.0000 - 6s/epoch - 20ms/step
Epoch 7/10
282/282 - 5s - loss: 0.0019 - accuracy: 1.0000 - 5s/epoch - 19ms/step
Epoch 8/10
282/282 - 5s - loss: 0.0013 - accuracy: 1.0000 - 5s/epoch - 19ms/step
Epoch 9/10
282/282 - 6s - loss: 9.1983e-04 - accuracy: 1.0000 - 6s/epoch - 20ms/step
Epoch 10/10
282/282 - 6s - loss: 6.7788e-04 - accuracy: 1.0000 - 6s/epoch - 22ms/step
282/282 - 3s - loss: 5.3322e-04 - accuracy: 1.0000 - 3s/epoch - 10ms/step
Training accuracy: 100.00%
32/32 - 0s - loss: 0.5497 - accuracy: 0.8720 - 275ms/epoch - 9ms/step
Test accuracy: 87.20%
Sample review: The Glory was a masterpiece of storytelling and emotion. Every scene was beautifully crafted, and the performances were outstanding.
Predicted sentiment: POSITIVE (Positive Probability: 0.99)
==================================================
Sample review: I couldn't connect with the characters in The Glory. The plot felt disjointed and confusing.
Predicted sentiment: NEGATIVE (Positive Probability: 0.94)
==================================================
Sample review: This movie had me on the edge of my seat from start to finish. The suspense and twists kept me guessing.
Predicted sentiment: POSITIVE (Positive Probability: 0.98)
==================================================
Sample review: The Glory was a visual delight, with stunning cinematography and breathtaking scenery.
Predicted sentiment: POSITIVE (Positive Probability: 0.86)
==================================================
Sample review: The dialogue in this movie felt forced and unnatural. It was hard to believe the characters' interactions.
Predicted sentiment: NEGATIVE (Positive Probability: 0.84)
==================================================
Sample review: I was moved to tears by the heartfelt performances in The Glory. The actors truly brought their characters to life.
Predicted sentiment: POSITIVE (Positive Probability: 0.98)
==================================================
Sample review: The pacing of the movie was slow and dragged on. I found myself losing interest in the story.
Predicted sentiment: NEGATIVE (Positive Probability: 0.96)
==================================================
Sample review: The Glory had a thought-provoking storyline that stayed with me long after the credits rolled.
Predicted sentiment: NEGATIVE (Positive Probability: 0.79)
==================================================
Sample review: The special effects in this movie were lackluster and took away from the overall experience.
Predicted sentiment: NEGATIVE (Positive Probability: 0.77)
==================================================
Sample review: The chemistry between the lead actors was palpable, adding depth to the romantic storyline in The Glory.
Predicted sentiment: POSITIVE (Positive Probability: 0.63)
==================================================
Sample review: The plot twists in this movie were so unexpected that I couldn't help but be impressed by the writing.
Predicted sentiment: POSITIVE (Positive Probability: 0.51)
==================================================
Sample review: Unfortunately, the plot holes in The Glory were glaring and made it difficult to fully enjoy the film.
Predicted sentiment: NEGATIVE (Positive Probability: 0.76)
==================================================
Sample review: The soundtrack elevated the emotions of The Glory, creating an immersive experience for the audience.
Predicted sentiment: POSITIVE (Positive Probability: 0.92)
==================================================
Sample review: The character development in this movie was shallow, leaving me feeling disconnected from their journeys.
Predicted sentiment: NEGATIVE (Positive Probability: 0.97)
==================================================
Sample review: The Glory was a rollercoaster of emotions, taking me from laughter to tears in a matter of scenes.
Predicted sentiment: POSITIVE (Positive Probability: 0.96)
==================================================
Sample review: The cinematography was top-notch, capturing the essence of each location in The Glory.
Predicted sentiment: POSITIVE (Positive Probability: 0.85)
==================================================
Sample review: The lack of diversity in the cast of The Glory was disappointing and didn't accurately reflect the real world.
Predicted sentiment: NEGATIVE (Positive Probability: 0.72)
==================================================
Sample review: The climax of the movie left me on the edge of my seat, and the resolution was satisfying and heartwarming.
Predicted sentiment: POSITIVE (Positive Probability: 0.88)
==================================================
Sample review: The dialogue flowed naturally, adding authenticity to the interactions between characters in The Glory.
Predicted sentiment: NEGATIVE (Positive Probability: 0.79)
==================================================
Sample review: Unfortunately, the acting in The Glory was wooden and lacked the emotional depth needed to fully engage the audience.
Predicted sentiment: NEGATIVE (Positive Probability: 0.78)
==================================================




	MCP
	CNN 
	Train Accuracy
	100%
	100%
	Test Accuracy
	87.20%
	87.40%
	















B. Embedding and CNN
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 1480, 100)         6395900   
                                                                 
 conv1d (Conv1D)             (None, 1473, 32)          25632     
                                                                 
 max_pooling1d (MaxPooling1D  (None, 736, 32)          0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 23552)             0         
                                                                 
 dense (Dense)               (None, 10)                235530    
                                                                 
 dense_1 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 6,657,073
Trainable params: 6,657,073
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
250/250 - 28s - loss: 0.5168 - accuracy: 0.6955 - 28s/epoch - 113ms/step
Epoch 2/10
250/250 - 31s - loss: 0.1364 - accuracy: 0.9528 - 31s/epoch - 124ms/step
Epoch 3/10
250/250 - 46s - loss: 0.0163 - accuracy: 0.9958 - 46s/epoch - 184ms/step
Epoch 4/10
250/250 - 46s - loss: 0.0028 - accuracy: 0.9996 - 46s/epoch - 184ms/step
Epoch 5/10
250/250 - 49s - loss: 9.6064e-04 - accuracy: 0.9999 - 49s/epoch - 196ms/step
Epoch 6/10
250/250 - 49s - loss: 2.5406e-04 - accuracy: 1.0000 - 49s/epoch - 197ms/step
Epoch 7/10
250/250 - 49s - loss: 1.4924e-04 - accuracy: 1.0000 - 49s/epoch - 196ms/step
Epoch 8/10
250/250 - 49s - loss: 9.9263e-05 - accuracy: 1.0000 - 49s/epoch - 195ms/step
Epoch 9/10
250/250 - 49s - loss: 7.0688e-05 - accuracy: 1.0000 - 49s/epoch - 197ms/step
Epoch 10/10
250/250 - 51s - loss: 5.2190e-05 - accuracy: 1.0000 - 51s/epoch - 202ms/step
Test Accuracy: 87.40%
Train Accuracy: 100.00%
Review: [The Glory was a masterpiece of storytelling and emotion. Every scene was beautifully crafted, and the performances were outstanding.]
Sentiment: POSITIVE (96.925%)
==================================================
Review: [I couldn't connect with the characters in The Glory. The plot felt disjointed and confusing.]
Sentiment: NEGATIVE (98.060%)
==================================================
Review: [This movie had me on the edge of my seat from start to finish. The suspense and twists kept me guessing.]
Sentiment: POSITIVE (99.760%)
==================================================
Review: [The Glory was a visual delight, with stunning cinematography and breathtaking scenery.]
Sentiment: POSITIVE (98.300%)
==================================================
Review: [The dialogue in this movie felt forced and unnatural. It was hard to believe the characters' interactions.]
Sentiment: NEGATIVE (93.452%)
==================================================
Review: [I was moved to tears by the heartfelt performances in The Glory. The actors truly brought their characters to life.]
Sentiment: POSITIVE (99.754%)
==================================================
Review: [The pacing of the movie was slow and dragged on. I found myself losing interest in the story.]
Sentiment: NEGATIVE (99.972%)
==================================================
Review: [The Glory had a thought-provoking storyline that stayed with me long after the credits rolled.]
Sentiment: POSITIVE (80.695%)
==================================================
Review: [The special effects in this movie were lackluster and took away from the overall experience.]
Sentiment: NEGATIVE (95.627%)
==================================================
Review: [The chemistry between the lead actors was palpable, adding depth to the romantic storyline in The Glory.]
Sentiment: POSITIVE (79.040%)
==================================================
Review: [The plot twists in this movie were so unexpected that I couldn't help but be impressed by the writing.]
Sentiment: NEGATIVE (86.924%)
==================================================
Review: [Unfortunately, the plot holes in The Glory were glaring and made it difficult to fully enjoy the film.]
Sentiment: POSITIVE (94.649%)
==================================================
Review: [The soundtrack elevated the emotions of The Glory, creating an immersive experience for the audience.]
Sentiment: POSITIVE (99.716%)
==================================================
Review: [The character development in this movie was shallow, leaving me feeling disconnected from their journeys.]
Sentiment: NEGATIVE (99.986%)
==================================================
Review: [The Glory was a rollercoaster of emotions, taking me from laughter to tears in a matter of scenes.]
Sentiment: POSITIVE (98.491%)
==================================================
Review: [The cinematography was top-notch, capturing the essence of each location in The Glory.]
Sentiment: POSITIVE (98.047%)
==================================================
Review: [The lack of diversity in the cast of The Glory was disappointing and didn't accurately reflect the real world.]
Sentiment: NEGATIVE (98.522%)
==================================================
Review: [The climax of the movie left me on the edge of my seat, and the resolution was satisfying and heartwarming.]
Sentiment: POSITIVE (96.538%)
==================================================
Review: [The dialogue flowed naturally, adding authenticity to the interactions between characters in The Glory.]
Sentiment: POSITIVE (93.646%)
==================================================
Review: [Unfortunately, the acting in The Glory was wooden and lacked the emotional depth needed to fully engage the audience.]
Sentiment: NEGATIVE (87.271%)
==================================================