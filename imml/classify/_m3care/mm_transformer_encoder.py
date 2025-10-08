# License: BSD-3-Clause

from .nmt import PositionalEncoding, MultiHeadedAttention, SublayerConnection, PositionwiseFeedForward

try:
    from torch import nn
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

nnModuleBase = nn.Module if deepmodule_installed else object


class MM_transformer_encoder(nnModuleBase):
    def __init__(self, input_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(MM_transformer_encoder, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        # self.d_k = d_k
        # self.d_v = d_v # the two can be equal
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.embed = nn.Linear(self.input_dim, self.d_model)

        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout=0, max_len=5000)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)
        self.output = nn.Linear(self.d_model, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # self.sparsemax = Sparsemax(dim=0)

    def forward(self, input, mask):
        # input shape [batch_size, timestep, feature_dim]
        #         demographic = demo_input

        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert (feature_dim == self.input_dim)
        assert (self.d_model % self.MHD_num_head == 0)

        input = self.embed(input)
        #         print(input.shape)
        input = self.PositionalEncoding(input)  # b t d_model
        #         posi_input = embed_input# b t d_model

        #         mask = subsequent_mask(time_step).to(device) # 1 t t 下三角
        #         print(mask)
        contexts = self.SublayerConnection(input, lambda x: self.MultiHeadedAttention(input, input, input,
                                                                                      mask))  # b t d_model
        # contexts = self.MultiHeadedAttention(qs, ks, vs, mask)# b t h

        contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))  # b t d_model

        #         contexts = contexts.view(batch_size, 16 * time_step)

        #
        #         os = []
        #         for j in range(contexts.shape[0]):
        #             os.append(contexts[j,lens[j]-1])

        #         os = torch.stack(os)

        #         output = self.output(os)# b t 1
        #         output = self.sigmoid(output)

        return contexts  # b t h