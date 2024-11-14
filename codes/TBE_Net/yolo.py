import torch
import torch.nn as nn
import copy


loss_fun = nn.TripletMarginWithDistanceLoss(
    margin=0.3, distance_function=nn.PairwiseDistance())
embedding = nn.Embedding(1000, 128)
anchor_ids = torch.randint(0, 1000, (1,))
positive_ids = torch.randint(0, 1000, (1,))
negative_ids = torch.randint(0, 1000, (1,))
anchor = embedding(anchor_ids)
positive = embedding(positive_ids)
negative = embedding(negative_ids)
