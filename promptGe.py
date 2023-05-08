import torch
import torch.nn as nn

class promptGe(nn.Module):
    def __init__(self, length=1, embed_dim=768, batchwise_prompt=False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.batchwise_prompt = batchwise_prompt


        #Generation network
        self.generation_layer_1 = nn.Linear(768, 256, bias=True)
        self.generation_activation = nn.Relu()
        self.generation_layer_2 = nn.Linear(256, 768*5, bias=True)

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()

        self.task_embed_layer = nn.Embedding(10, self.embed_dim)
        self.task_embed = torch.zeros(1, 1, self.embed_dim)
        m = self.task_embed.expand(x_embed.shape[0], -1, -1)
        n = self.task_embed_layer(m)

        x_task_embed = torch.cat((x_embed, n), dim=1)
        a = self.generation_layer_1(x_task_embed)
        b = self.generation_activation(a)
        c = self.generation_layer_2(b)

        