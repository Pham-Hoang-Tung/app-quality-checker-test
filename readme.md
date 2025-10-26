reasoning_net

Typical mlp net:
    
    text -> embedding -> mlp -> score

    Problem:
        Probably overfitting like crazy


Experimental:
    Idea : Add reasoning module to force model to learn causal relationships

        text -> encoder -> reasoning module -> score module -> score
        
    Problem:
        Reasoning texts are noisy.
        At worst, they are garbage, you get a mlp with extra layers that actively hurt training.
        
To do:
    - Try different reasoning text generation methods
    - Drop reasoning and try contrastive learning first, but this one is very manual so maybe not. 