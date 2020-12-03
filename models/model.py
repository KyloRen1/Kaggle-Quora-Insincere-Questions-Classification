import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            embedding_matrix,
            vocab_size,
            embedding_dim=300,
            use_pretrained=False,
            dropout=0.25):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        if use_pretrained:
            self.embedding.weight.data.copy_(
                torch.from_numpy(embedding_matrix))

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        return x


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )

    def init_weights(self):
        ih = (
            param.data for name,
            param in self.named_parameters() if 'weight_ih' in name)
        hh = (
            param.data for name,
            param in self.named_parameters() if 'weight_hh' in name)
        b = (
            param.data for name,
            param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        x = self.gru(x)
        return x


class CapsulLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            num_capsule,
            dim_capsule,
            routings=4,
            kernel_size=(
                9,
                1),
            share_weights=True,
            activation='default',
            **kwargs):
        super(CapsulLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(
                    1,
                    input_dim,
                    self.num_capsule *
                    self.dim_capsule)))

    def forward(self, x):
        u_hat_vector = torch.matmul(x, self.W)
        batch_size, num_capsule, _ = x.size()
        u_hat_vector = u_hat_vector.view(
            (batch_size, num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vector = u_hat_vector.permute(0, 2, 1, 3)
        b = torch.zeros_like(u_hat_vector[:, :, :, 0])

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum(
                'bij,bijk->bik', (c, u_hat_vector)))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vector))
        return outputs

    def squash(self, x, axis=-1, epsilon=1e-7):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + epsilon)
        return x / scale


class DenseLayer(nn.Module):
    def __init__(self, input_shape, output_shape, dropout=0.25):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(input_shape, output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class CapsulLayerMain(nn.Module):
    def __init(self, embedding_matrix, vocab_size, input_size, hidden_size):
        super(CapsulLayerMain, self).__init__()
        self.embedding = EmbeddingLayer(embedding_matrix, vocab_size)
        self.gru = GRULayer(input_size, hidden_size)
        self.gru.init_weights()
        self.capsule = CapsulLayer(hidden_size * 2, num_capsule, dim_capsule)
        self.dense = DenseLayer(num_capsule * dim_capsule, 30)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.capsule(x)
        x = self.dense(x)
        return


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.support_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        self.epsilon = 1e-10

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim),
                       self.weight).view(-1, self.step_dim)
        if self.bias:
            eij = eij + self.b
        eij - torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + self.epsilon
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_shape,
            embedding_matrix,
            hidden_size,
            linear_hidden_size,
            num_capsule,
            dim_capsule,
            embedding_dropout=0.1,
            dropout=0.1,
            max_len=70):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_shape)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_shape,
            hidden_size,
            bidirectional=True,
            batch_first=True)
        self.gru = nn.GRU(
            hidden_size * 2,
            hidden_size,
            bidirectional=True,
            batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            bidirectional=True,
            batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, max_len)
        self.gru_attention = Attention(hidden_size * 2, max_len)

        self.capsule = CapsulLayer(hidden_size * 2, num_capsule, dim_capsule)

        self.relu = nn.ReLU()
        self.embedding_dropout = nn.Dropout2d(embedding_dropout)
        self.dropout = nn.Dropout(dropout)

        self.linear_capsule = nn.Linear(num_capsule * dim_capsule, 1)
        self.linear = nn.Linear(hidden_size * 8 + 3, linear_hidden_size)
        self.out = nn.Linear(linear_hidden_size, 1)

    def forward(self, x):
        h_embedding = self.embedding(x[0])
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0))
        )

        output_lstm, _ = self.lstm(h_embedding)
        output_gru, _ = self.gru(output_lstm)

        content = self.capsule(output_gru)
        content = self.dropout(content)
        batch_size = content.size(0)
        content = content.view(batch_size, -1)
        content = self.relu(self.linear_capsule(content))

        output_lstm_attention = self.lstm_attention(output_lstm)
        output_gru_attention = self.gru_attention(output_gru)

        average_pooling = torch.mean(output_gru, 1)
        max_pooling, _ = torch.max(output_gru, 1)

        f = torch.tensor(x[1], dtype=torch.float).to(device)

        concat = torch.cat(
            (output_lstm_attention,
             output_gru_attention,
             content,
             average_pooling,
             max_pooling,
             f),
            1)
        concat = self.relu(self.linear(concat))
        concat = self.dropout(concat)
        output = self.out(concat)
        return output
