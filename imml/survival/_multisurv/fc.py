from _bisect import bisect_left

from ... import Module, deepmodule_installed

if deepmodule_installed:
    from torch import nn

class FC(Module):

    def __init__(self, in_features, out_features, n_layers, dropout=True,
                 batchnorm=False, scaling_factor=4):
        super().__init__()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout, batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(in_features, n_neurons, dropout, batchnorm=True)
                layers += self._make_layer(n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(in_features, n_neurons, dropout, batchnorm=True)
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(n_in, n_neurons, dropout, batchnorm=True)
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm)

        self.fc = nn.Sequential(*layers)


    def _make_layer(self, in_features, out_features, dropout, batchnorm):
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout())
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer


    def _pick_n_neurons(self, n_features):
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)
        return n_neurons[0 if idx == 0 else idx - 1]


    def forward(self, x):
        return self.fc(x)
