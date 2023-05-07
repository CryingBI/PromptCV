import torch
import torch.nn as nn

class promptGe(nn.Module):
    def __init__(self, length=5, embed_dim=768, top_k=None, batchwise_prompt=False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        
        # self.prompt_pool = prompt_pool
        # self.embedding_key = embedding_key
        # self.prompt_init = prompt_init
        # self.prompt_key = prompt_key
        # self.pool_size = pool_size

        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.task_embed = None
        task_len = self.task_embed.shape[1]
        self.generation_layer_1 = nn.Linear(768, 256, bias=True)
        self.generation_activation = nn.Relu()
        self.generation_layer_2 = nn.Linear(256, 768*5, bias=True)

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()

        x_task_embed = torch.cat((x_embed, self.task_embed), dim=1)
        a = self.generation_layer_1(x_task_embed)
        b = self.generation_activation(a)
        c = self.generation_layer_2(b)

        